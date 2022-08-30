import tensorflow as tf
from tensorflow.keras.layers import Layer

from model.transformer.encoder_block import EncoderBlock

class NotebookTransformer(Layer):
    def __init__(self, d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers):
        """
        Args:
            d_model (int): Dimensionality of the feature embedding
            n_heads (int): The number of heads for the multi-head attention
            dropout (float): Dropout rate
            eps (float): Epsilon for layer normalization
            is_training (bool): Whether the model is training
            d_ff (int): Dimensionality of the feed forward layer
            ff_activation (str): Activation function of the feed forward layer
            n_layers (int): Number of transformer encoder blocks to be stacked
        """
        
        super(NotebookTransformer, self).__init__()
        self.encoder_layers = self.encoder_layers = [EncoderBlock(d_model, n_heads, dropout, eps, d_ff, ff_activation) for _ in range(n_layers)]


    def call(self, x, is_training, mask=None):
        """
        Args:
            x (tensor): Input with shape (..., num_cells, d_model)
            is_training (bool): Whether the model is being trained
            mask (tensor): Mask with shape (..., seqlen)
        Returns:
            x (tensor): Output with shape (..., num_cells, d_model)
        """

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, is_training, mask)
        return x
