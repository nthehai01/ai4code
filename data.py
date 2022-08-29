import tensorflow as tf
from transformers import AutoTokenizer
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path

class Dataset:
    """
    Data preparator for the model.
    """

    def __init__(self, data_dir, model_path, max_len, num_train, buffer_size, batch_size):
        """
        Args:
            data_dir (str): Path to the data directory.
            model_path (str): Path of the pre-trained model.
            max_len (int): Maximum length of a sentence.
            num_train (int): Number of notebook to be used for training.
            buffer_size (int): Buffer size for shuffling.
            batch_size (int): Batch size.
        """
        
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.num_train = num_train
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.buffer_size = buffer_size
        self.batch_size = batch_size


    def read_notebook(self, path):
        """
        Read a notebook.

        Args: 
            path (str): Path to this notebook.
        Returns:
            content (pd): Notebook content.
        """

        return (
            pd.read_json(
                path,
                dtype={'cell_type': 'category', 
                    'source': 'str'}
            )
            .assign(id=path.stem)
            .rename_axis('cell_id')
        )

    
    def read_notebooks(self):
        """
        Read the notebooks to be used for training.

        Returns:
            df (pd): Training dataset.
        """

        paths_train = list((self.data_dir / 'train').glob('*.json'))[:self.num_train]
        notebooks_train = [
            self.read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
        ]

        df = (
            pd.concat(notebooks_train)
            .set_index('id', append=True)
            .swaplevel()
            .sort_index(level='id', sort_remaining=False)
        )

        return df

    
    def add_ranking(self, df):
        """
        Add the ranking of each cell within a notebook.

        Args:
            df (pd): Dataset.
        Returns:
            df (pd): Dataset with ranking in the ascending order of each notebook.
        """

        def get_ranks(base, derived):
            return [base.index(d) for d in derived]

        df_orders = pd.read_csv(
            self.data_dir / 'train_orders.csv',
            index_col='id',
            squeeze=True,
        ).str.split()

        # Create a ranking dataframe for the whole notebooks
        df_orders_ = df_orders.to_frame().join(
            df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
            how='right',
        )

        ranks = {}
        for id_, cell_order, cell_id in df_orders_.itertuples():
            ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

        df_ranks = (
            pd.DataFrame
            .from_dict(ranks, orient='index')
            .rename_axis('id')
            .apply(pd.Series.explode)
            .set_index('cell_id', append=True)
        )
        
        # Merge Ranking to the main dataframe
        df = df.reset_index().merge(df_ranks, on=["id", "cell_id"])

        # Add percentile rank
        df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

        # Sort the dataframe by the ranking
        df = df.sort_values(by=['id', 'rank']).reset_index(drop=True)

        return df


    def tokenize(self, sentence):
        """
        Tokenize a sentence.
        
        Args:
            sentence (str): A sentence.
        Returns:
            tokens (tuple): A list of input ids and attention mask.
        """
        
        tokens = self.tokenizer.encode_plus(
            sentence, 
            max_length=self.max_len,
            truncation=True, 
            padding='max_length',
            add_special_tokens=True, 
            return_attention_mask=True,
            return_token_type_ids=False, 
            return_tensors='tf'
        )
        
        return tokens['input_ids'], tokens['attention_mask']

    
    def clean_text(self, text):
        """
        Make the text cleaner.

        Args:
            text (str): text.
        Returns:
            text (str): A cleaned text.
        """
        
        text = text.lower().strip()
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = text.strip()

        return text


    def preprocess_content(self, cell):
        """
        Preprocess the content of a cell by clean source, tokenizing and getting features.

        Args:
            content (str): The cell.
        Returns:
            content (str): Preprocessed content.
        """

        # Clean the source
        cell.source = self.clean_text(cell.source)

        # Tokenize the source
        input_ids, attention_mask = self.tokenize(cell.source)
        cell.input_ids = input_ids.numpy()[0]
        cell.attention_mask = attention_mask.numpy()[0]

        # Get the cell features
        cell_features = [int(cell.cell_type == 'code'), cell.pct_rank]
        cell.cell_features = cell_features

        return cell


    def load_dataset(self):
        """
        Load the dataset for training.

        Returns:
            df (pd): Training dataset.
        """

        df = self.read_notebooks()
        df = self.add_ranking(df)

        return df

    
    def preprocess_dataset(self, df):
        """
        Preprocess the dataset for training.

        Args:
            df (pd): Dataset.
        Returns:
            dataset (pd): Preprocessed dataset.
        """

        df['input_ids'] = 0
        df['attention_mask'] = 0
        df['cell_features'] = 0
        df = df.apply(self.preprocess_content, axis=1)

        df = df.drop(['cell_id', 'cell_type', 'source', 'rank'], axis=1)

        return df


    def build_dataset(self):
        """
        Build the dataset for training.
        """

        def map_func(input_ids, attention_mask, cell_features, pct_rank):
            return ( 
                {
                    'input_ids': input_ids, 
                    'attention_mask': attention_mask, 
                    'cell_features': cell_features
                }, 
                pct_rank 
            )

        df = self.load_dataset()
        df = self.preprocess_dataset(df)

        dataset = tf.data.Dataset.from_tensor_slices((
            df['input_ids'].tolist(), 
            df['attention_mask'].tolist(), 
            df['cell_features'].tolist(), 
            df['pct_rank'].tolist()
        ))
        dataset = dataset.map(map_func)

        return dataset.shuffle(self.buffer_size).batch(self.batch_size)
