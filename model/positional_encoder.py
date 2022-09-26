import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout


class PositionalEncoder(Layer):
    def __init__(self, d_model, num_cells, dropout):
        """
        Args:
            d_model (int): Dimensionality of the model
            num_cells (int): Maximum number of cells allowed for a notebook.
            dropout (float): Dropout rate
        """

        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.num_cells = num_cells
        self.dropout = Dropout(dropout)
        

    def call(self, x, is_training):
        """
        Perform positional encoding.
        
        Args:
            x (tensor): Input with shape (..., num_cells, d_model)
        Returns:
            out (tensor): Output with shape (..., num_cells, d_model)
        """
        
        def createAngleRates(d_model):
            angles = np.arange(d_model)
            angles[1::2] = angles[0::2]
            angles = 1 / (10000 ** (angles / d_model))
            angles = np.expand_dims(angles, axis=0)
            return angles


        def generate_positional_encoding(pos, d_model):
            angles = createAngleRates(d_model)
            pos = np.expand_dims(np.arange(pos), axis=1)
            pos_angles = pos.dot(angles)
            pos_angles[:, 0::2] = np.sin(pos_angles[:, 0::2])
            pos_angles[:, 1::2] = np.cos(pos_angles[:, 1::2])
            pos_angles = np.expand_dims(pos_angles, axis=0)

            return tf.cast(pos_angles, dtype=tf.float32)

       
        pos_encoding = generate_positional_encoding(self.num_cells, self.d_model)
        
        out = x + pos_encoding[:, :self.num_cells, :]

        out = self.dropout(out, training=is_training)

        return out