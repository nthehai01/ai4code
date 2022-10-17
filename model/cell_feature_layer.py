import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class CellFeatureLayer(Layer):
    def __init__(self, d_model):
        """
        Args:
            cell_type (str): cell type (code -> 1, markdown -> 0)
            cell_pos (str): percentile rank of the cell
        """
        
        super(CellFeatureLayer, self).__init__()
        self.d_model = d_model
        self.ff = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(256, activation='relu'),
            Dense(d_model)
        ])
    

    # def call(self, cell_type, cell_pos):
    #     """
    #     Args:
    #         cell_type (tensor): cell types with shape (..., num_cells, 1)
    #         cell_pos (tensor): percentile rank of the cells with shape (..., num_cells, 1)
    #     Returns:
    #         out (tensor): cell feature embedding with shape (..., num_cells, 1, d_model)
    #     """
        
    #     batch_size = tf.shape(cell_type)[0]
    #     num_cells = tf.shape(cell_type)[1]

    #     out = tf.concat([cell_type, cell_pos], axis=-1)
    #     out = self.ff(out)
    #     out = tf.reshape(out, (batch_size, num_cells, 1, -1))
    #     return out

    def call(self, cell_features):
        """
        Args:
            cell_features (tensor): Cell features with shape (..., num_cells, 2)
        Returns:
            out (tensor): cell feature embedding with shape (..., num_cells, 1, d_model)
        """
        
        batch_size = tf.shape(cell_features)[0]
        num_cells = tf.shape(cell_features)[1]

        out = self.ff(cell_features)
        out = tf.reshape(out, (batch_size, num_cells, 1, self.d_model))
        return out
        