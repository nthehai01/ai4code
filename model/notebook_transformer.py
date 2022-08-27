import tensorflow as tf
from tensorflow.keras.layers import Layer

from model.transformer.encoder_block import EncoderBlock

class NotebookTransformer(Layer):
    def __init__(self, d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers):
        """
        Args:
            d_model (int): dimentionality of the feature embedding
            n_heads (int): the number of heads for the multi-head attention
            dropout (float): dropout rate
            eps (float): epsilon for layer normalization
            is_training (bool): whether the model is training
            d_ff (int): dimentionality of the feed forward layer
            ff_activation (str): activation function of the feed forward layer
            n_layers (int): number of transformer encoder blocks to be stacked
        """
        
        super(NotebookTransformer, self).__init__()
        self.encoder_layers = self.encoder_layers = [EncoderBlock(d_model, n_heads, dropout, eps, d_ff, ff_activation) for _ in range(n_layers)]


    def call(self, x, is_training, mask=None):
        """
        Args:
            x (tensor): input with shape (..., num_cells, d_model)
            is_training (bool): whether the model is being trained
            mask (tensor): mask with shape (..., seqlen)
        Returns:
            x (tensor): output with shape (..., num_cells, d_model)
        """

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, is_training, mask)
        return x
