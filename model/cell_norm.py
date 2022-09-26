import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization

class CellNorm(Layer):
    def __init__(self, d_model):
        super(CellNorm, self).__init__()
        self.d_model = d_model
        self.batch_norm = BatchNormalization()

    def call(self, x):
        """
        Args:
            x (tensor): Input with shape (..., num_cells, max_len, d_model)
        Returns:
            x (tensor): Output with shape (..., num_cells, max_len, d_model)
        """
        
        num_cells = tf.shape(x)[1]
        max_len = tf.shape(x)[2]

        x = tf.reshape(x, shape=(-1, max_len, self.d_model))
        x = self.batch_norm(x)
        x = tf.reshape(x, shape=(-1, num_cells, max_len, self.d_model))
        return x 