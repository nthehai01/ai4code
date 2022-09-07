import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization

class CellNorm(Layer):
    def __init__(self, **kwargs):
        super(CellNorm, self).__init__()
        self.batch_norm = BatchNormalization()

    def call(self, x):
        """
        Args:
            x (tensor): Input with shape (..., num_cells, max_cell, d_model)
        Returns:
            x (tensor): Output with shape (..., num_cells, max_cell, d_model)
        """
        x = self.batch_norm(x)
        return x
