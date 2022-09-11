import tensorflow as tf

from model.cell_transformer import CellTransformer
from model.cell_feature_layer import CellFeatureLayer
# from model.cell_norm import CellNorm
from model.attention_pooling import AttentionPooling
from model.notebook_transformer import NotebookTransformer
from model.pointwise_head import PointwiseHead

class Model(tf.keras.Model):
    def __init__(self, model_path, d_model, d_ff_pool, 
                n_heads, dropout_trans, eps, d_ff_trans, ff_activation, n_layers,
                d_ff_pointwise, dropout_pointwise):
        """
        Args:
            model_path (str): Path of the pre-trained model.
            d_model (int): Dimension of the model.
            d_ff_pool (int): Dimension of the feed forward layer for the AttentionPooling.
            n_heads (int): The number of heads for the multi-head attention.
            dropout_trans (float): Dropout rate for the NotebookTransformer.
            eps (float): Epsilon for layer normalization.
            d_ff_trans (int): Dimension of the feed forward layer.
            ff_activation (str): Activation function of the feed forward layer.
            n_layers (int): Number of transformer encoder blocks to be stacked.
            d_ff_pointwise (int): Dimension of the feed forward layer for the PointwiseHead.
            dropout_pointwise (float): Dropout rate for the PointwiseHead.
        """
        
        super(Model, self).__init__()
        self.cell_transformer = CellTransformer(model_path, d_model)
        self.cell_feature_layer = CellFeatureLayer(d_model)
        # self.cell_norm = CellNorm()
        self.attention_pooling = AttentionPooling(d_ff_pool, d_model)
        self.notebook_transformer = NotebookTransformer(d_model, n_heads, dropout_trans, eps, d_ff_trans, ff_activation, n_layers)
        self.pointwise_head = PointwiseHead(d_ff_pointwise, dropout_pointwise)


    def call(self, input_ids, attention_mask, cell_features, cell_mask, is_training=False):
        """
        Args:
            input_ids (tensor): List of the input IDs of the tokens with shape (..., num_cells, max_len)
            attention_mask (tensor): List of the attention masks of the tokens with shape (..., num_cells, max_len)
            cell_features (tensor): Cell features with shape (..., num_cells, 2)
            cell_mask (tensor): Cell mask with shape (..., num_cells)
            is_training (bool): Whether the model is being trained. Default: False.
        Returns:
            out (tensor): Output with shape (..., num_cells)
        """
        
        embeddings = self.cell_transformer(input_ids, attention_mask)  # shape (..., num_cells, max_len, d_model)
        features = self.cell_feature_layer(cell_features)  # shape (..., num_cells, 1, d_model)

        out = tf.concat([embeddings, features], axis=-2)  # shape (..., num_cells, max_len + 1, d_model)

        # out = self.cell_norm(out)  # shape (..., num_cells, max_len + 1, d_model)
        
        out = self.attention_pooling(out)  # shape (..., num_cells, d_model)

        out = self.notebook_transformer(out, is_training)  # shape (..., num_cells, d_model)

        out = self.pointwise_head(out, cell_mask, is_training)  # shape (..., num_cells)

        return out
