import tensorflow as tf
from transformers import AutoTokenizer
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

class Dataset:
    """
    Data preparator for the model.
    """

    def __init__(self, data_dir, model_path, max_len, num_cells, num_train, buffer_size, batch_size, cell_pad):
        """
        Args:
            data_dir (str): Path to the data directory.
            model_path (str): Path of the pre-trained model.
            max_len (int): Maximum length of a sentence.
            num_cells (int): Maximum number of cells allowed for a notebook.
            num_train (int): Number of notebook to be used for training, load all if -1.
            buffer_size (int): Buffer size for shuffling.
            batch_size (int): Batch size.
            cell_pad (int): Pad value for the fake cell (padded cell).
        """
        
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.num_cells = num_cells
        self.num_train = num_train
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.cell_pad = cell_pad


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

        if self.num_train == -1:
            paths_train = list((self.data_dir / 'train').glob('*.json'))
            self.num_train = len(paths_train)
        else:
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


    def add_ancestors(self, df):
        """
        Add the ranking of each cell within a notebook.

        Args:
            df (pd): Dataset.
        Returns:
            df (pd): Dataset with ancestor info.
        """

        df_ancestors = pd.read_csv(self.data_dir / 'train_ancestors.csv', index_col='id')
        
        df = df.reset_index(drop=True).merge(df_ancestors, on=["id"])
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
        cell.input_ids = input_ids.numpy()[0].tolist()
        cell.attention_mask = attention_mask.numpy()[0].tolist()

        # Get the cell features
        cell_features = [
            int(cell.cell_type == 'code'), 
            cell.pct_rank if cell.cell_type == 'code' else 0
        ]
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
        df = self.add_ancestors(df)

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

        df = df.drop(['cell_type', 'source', 'rank'], axis=1)

        return df


    def filter_by_num_cells(self, df, max_cells, min_cells=0):
        """
        Filter the notebooks by the number of cells containing.

        Args:
            df (pd): Notebook dataframe.
            max_cells (int): The upper bound on the number of cells to keep.
            min_cells (int): The lower bound on the number of cells to keep.
        Returns:
            filtered_df (pd): The dataframe after being filtered.
        """

        cell_count = df.groupby("id").count()
        cell_count = cell_count["cell_id"]
        temp = cell_count[(cell_count >= min_cells) & (cell_count <= max_cells)]

        filtered_df = df[df['id'].isin(temp.keys())]
        return filtered_df


    def get_notebook_token(self, df):
        """
        Get the tokens for model. In this function, we'll pad the notebooks to be equal in term of number of cells that a notebook can maximum contains (i.e. This process is pretty much the same compared to the way we pad for a single sentence before). The returned cell_mask will tell us whether it's actually a real cell (real cell -> 1) or a padded version (fake cell -> 0).

        Args:
            df (pd): The tokenized notebook dataframe which contains the tokens instead of the rough content for each cell.
        Returns:
            input_ids (np array): Input ids with shape (num_notebooks, num_cells, max_len)
            attention_mask (np array): Attention mask with shape (num_notebooks, num_cells, max_len)
            cell_features (np array): Cell features with shape (num_notebooks, num_cells, 2)
            cell_mask (np array): Cell mask with shape (num_notebooks, num_cells, 1)
            target (np array): Percentile rank with shape (num_notebooks, num_cells, 1)
        """

        def create_tensor(col, desired_shape, dtype="int32"):
            """
            Create the desired tensor.

            Args:
                col (str): Column name needed to be tensorized.
                desired_shape (tuple): Desired output's shape.
                dtype (str): Data type. Default is int32.
            Returns:
                out (np array): Padded output with the shape of desired_shape.
            """

            out = np.full(shape=desired_shape, fill_value=self.cell_pad, dtype=dtype)
            
            count = 0
            for _, group in df.groupby("id"):
                value = group[col].tolist()
                value_shape = np.array(value).shape
                
                if len(value_shape) == 1:
                    out[count, :value_shape[0]] = value
                else:
                    out[count, :value_shape[0], :value_shape[1]] = value

                count += 1

            return out

        # input_ids
        input_ids = create_tensor(
            "input_ids", 
            (self.num_train, self.num_cells, self.max_len)
        )

        # attention_mask
        attention_mask = create_tensor(
            "attention_mask", 
            (self.num_train, self.num_cells, self.max_len)
        )

        # cell_features
        cell_features = create_tensor(
            "cell_features", 
            (self.num_train, self.num_cells, 2),
            dtype="float32"
        )

        # cell_mask
        cell_mask = np.zeros((self.num_train, self.num_cells, 1), dtype="float32")
        count = 0
        for _, group in df.groupby("id"):
            value = group["input_ids"].tolist()
            value_shape = np.array(value).shape
            cell_mask[count, :value_shape[0], :] = 1.
            count += 1

        # target
        target = create_tensor("pct_rank", (self.num_train, self.num_cells), dtype="float32")

        return input_ids, attention_mask, cell_features, cell_mask, target


    def save_token(self, token, file_path):
        """
        Save the token to the disk.

        Args:
            token (np array): The token needed to be saved.
            file_path (str): Path to save the token.
        """

        with open(file_path, 'wb') as f:
            np.save(f, token)
        

    def load_token(self, file_path):
        """
        Load the token from the disk.

        Args:
            file_path (str): Path to load the token.
        Returns:
            token (np array): The loaded token.
        """

        with open(file_path, 'rb') as f:
            token = np.load(f)

        return token
        

    def build_dataset(self, df=None, tensor_path=None):
        """
        Build the dataset for training.

        Args:
            df (pd): Notebook dataframe. If provided, the dataset will be using. Otherwise, the dataset will be loaded from the disk.
            tensor_path (str): Path to the main directory for saving the encoded arrays.
        """

        def map_func(input_ids, attention_mask, cell_features, cell_mask, target):
            return ( 
                {
                    'input_ids': input_ids, 
                    'attention_mask': attention_mask, 
                    'cell_features': cell_features,
                    'cell_mask': cell_mask
                }, 
                target 
            )

        
        if df is None:
            df = self.load_dataset()
            
        self.num_train = len(df)
        df = self.preprocess_dataset(df)
        df = self.filter_by_num_cells(df, self.num_cells)

        input_ids, attention_mask, cell_features, cell_mask, target = self.get_notebook_token(df)
        if tensor_path:
            self.save_token(input_ids, tensor_path + "/input_ids.npy")
            self.save_token(attention_mask, tensor_path + "/attention_mask.npy")
            self.save_token(cell_features, tensor_path + "/cell_features.npy")
            self.save_token(cell_mask, tensor_path + "/cell_mask.npy")
            self.save_token(target, tensor_path + "/target.npy")

        dataset = tf.data.Dataset.from_tensor_slices((
            input_ids, 
            attention_mask, 
            cell_features, 
            cell_mask,
            target
        ))
        dataset = dataset.map(map_func)

        return dataset.shuffle(self.buffer_size).batch(self.batch_size)
