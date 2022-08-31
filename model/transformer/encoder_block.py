import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization

from model.transformer.multi_head_attention import MultiHeadAttention

class EncoderBlock(Layer):
    def __init__(self, d_model, n_heads, dropout, eps, d_ff, ff_activation):
        """
        Args:
            d_model (int): Dimensionality of the feature embedding.
            n_heads (int): The number of heads for the multi-head attention.
            dropout (float): Dropout rate.
            eps (float): Epsilon for layer normalization.
            d_ff (int): Dimensionality of the feed forward layer.
            ff_activation (str): Activation function of the feed forward layer.
        """
        
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNormalization(epsilon=eps)
        self.norm2 = LayerNormalization(epsilon=eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.d_ff = d_ff
        self.ff_activation = ff_activation


    def feed_forward(self):
        """
        Perform a feed forward layer.
        
        Returns:
            ff (model): feed forward layer
        """
        
        return tf.keras.Sequential([
            Dense(self.d_ff, activation=self.ff_activation),
            Dense(self.d_model)
        ])


    def call(self, x, is_training, mask=None):
        """
        Perform an encoder block.

        Args:
            x (tensor): input with shape (..., seqlen, d_model)
            is_training (bool): whether the model is being trained
            mask (tensor): mask with shape (..., seqlen)
        Returns:
            x (tensor): output with shape (..., seqlen, d_model)
        """

        # Multi-head attention
        mha_output = self.mha(x, x, x, mask)
        mha_output = self.dropout1(mha_output, training=is_training)

        # Add & Norm
        x = x + mha_output
        x = self.norm1(x)

        # TODO: FIX AND UNFREEZE THESE FOLLOWING LINES OF CODE
        # # Feed forward
        # ff_output = self.feed_forward()(x)
        # ff_output = self.dropout2(ff_output, training=is_training)

        # # Add & Norm
        # x = x + ff_output
        # x = self.norm2(x)

        return x
