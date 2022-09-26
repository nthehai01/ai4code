# import tensorflow as tf
# from tensorflow.keras.layers import Layer, GlobalAveragePooling1D

# class AttentionPooling(Layer):
#     def __init__(self, d_ff, d_model):
#         """
#         Args:
#             d_ff (int): Dimensionality of the feed forward layer
#             d_model (int): Dimensionality of the model
#         """
        
#         super(AttentionPooling, self).__init__()
#         self.attention = GlobalAveragePooling1D()
#         self.d_model = d_model


#     def call(self, x):
#         """
#         Perform attention pooling.
        
#         Args:
#             x (tensor): Input with shape (..., num_cells, max_len + 1, d_model)
#         Returns:
#             out (tensor): Output with shape (..., num_cells, d_model)
#         """
        
#         num_cells = tf.shape(x)[1]
#         max_len_plus_1 = tf.shape(x)[2]
#         x = tf.reshape(x, (-1, max_len_plus_1, self.d_model))
        
#         out = self.attention(x)

#         out = tf.reshape(out, (-1, num_cells, self.d_model))

#         return out


import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling1D

class AttentionPooling(Layer):
    def __init__(self, d_model):
        """
        Args:
            d_model (int): Dimensionality of the model
        """
        
        super(AttentionPooling, self).__init__()
        self.attention = GlobalAveragePooling1D()
        self.d_model = d_model


    def call(self, x):
        """
        Perform attention pooling.
        
        Args:
            x (tensor): Input with shape (..., num_cells, max_len, d_model)
        Returns:
            out (tensor): Output with shape (..., num_cells, d_model)
        """
        
        num_cells = tf.shape(x)[1]
        max_len = tf.shape(x)[2]
        x = tf.reshape(x, shape=(-1, max_len, self.d_model))
        
        out = self.attention(x)

        out = tf.reshape(out, shape=(-1, num_cells, self.d_model))

        return out
