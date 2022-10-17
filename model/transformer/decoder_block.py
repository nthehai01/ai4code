import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization

from model.transformer.multi_head_attention import MultiHeadAttention

class DecoderBlock(Layer):
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
        
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNormalization(epsilon=eps)
        self.norm2 = LayerNormalization(epsilon=eps)
        self.norm3 = LayerNormalization(epsilon=eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.feed_forward = tf.keras.Sequential([
            Dense(d_ff, activation=ff_activation),
            Dense(self.d_model)
        ])


    def call(self, q, encoder_out, is_training, mask=None):
        """
        Perform an encoder block.

        Args:
            q (tensor): query with shape (..., seqlen, d_model)
            encoder_out (tensor): encoder output with shape (..., seqlen, d_model)
            is_training (bool): whether the model is being trained
            mask (tensor): mask with shape (..., seqlen)
        Returns:
            x (tensor): output with shape (..., seqlen, d_model)
        """

        mha_output = self.mha(q, q, q, mask)
        q = q + mha_output
        q = self.dropout1(q, training=is_training)
        q = self.norm1(q)

        k = v = encoder_out

        mha_output = self.mha(q, k, v, mask)
        q = q + mha_output
        mha_output = self.dropout2(mha_output, training=is_training)
        q = self.norm2(q)

        # Feed forward
        ff_output = self.feed_forward(q)
        ff_output = self.dropout3(ff_output, training=is_training)

        # Add & Norm
        q = q + ff_output
        q = self.norm3(q)

        return q
