import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class CellFeatureLayer(Layer):
    def __init__(self, d_model):
        """
        Args:
            cell_type (str): cell type (code -> 1, markdown -> 0)
            cell_pos (str): percentile rank of the cell
        """
        
        super(CellFeature, self).__init__()
        self.d_model = d_model
        self.ff = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(d_model, activation='relu')
        ])
    

    def call(self, cell_type, cell_pos):
        """
        Args:
            cell_type (tensor): cell types with shape (..., num_cells, 1)
            cell_pos (tensor): percentile rank of the cells with shape (..., num_cells, 1)
        Returns:
            out (tensor): cell feature embedding with shape (..., num_cells, d_model)
        """
        
        out = tf.concat([cell_type, cell_pos], axis=-1)
        out = self.ff(out)
        return out
        