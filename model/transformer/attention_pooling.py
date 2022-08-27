import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Softmax

class AttentionPooling(Layer):
    """
    This is inspired by the equation (3) in this paper: 
    https://aclanthology.org/2020.emnlp-main.511.pdf
    """
    def __init__(self, d_ff):
        """
        Args:
            d_ff (int): dimentionality of the feed forward layer
        """
        
        super(AttentionPooling, self).__init__()
        self.attention = tf.keras.Sequential([
            Dense(d_ff, activation='tanh'),
            Softmax(axis=-2),
            Dense(1)
        ])


    def call(self, x):
        """
        Perform attention pooling.
        
        Args:
            x (tensor): input with shape (..., num_cells, max_len + 1, d_model)
        Returns:
            out (tensor): output with shape (..., num_cells, d_model)
        """
        
        attention = self.attention(x)
        out = tf.reduce_sum(attention * x, axis=-2)
        return out
