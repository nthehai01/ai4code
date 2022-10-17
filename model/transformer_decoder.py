import tensorflow as tf
from tensorflow.keras.layers import Layer

from model.transformer.decoder_block import DecoderBlock

class TransformerDecoder(Layer):
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
        
        super(TransformerDecoder, self).__init__()
        self.decoder_layers = [DecoderBlock(d_model, n_heads, dropout, eps, d_ff, ff_activation) for _ in range(n_layers)]


    def call(self, q, encoder_out, is_training, mask=None):
        """
        Args:
            q (tensor): Input with shape (..., num_cells, d_model)
            encoder_out (tensor): Encoder output with shape (..., num_cells, d_model)
            is_training (bool): Whether the model is being trained
            mask (tensor): Mask with shape (..., seqlen)
        Returns:
            q (tensor): Output with shape (..., num_cells, d_model)
        """

        for decoder_layer in self.decoder_layers:
            q = decoder_layer(q, encoder_out, is_training, mask)
        return q