import tensorflow as tf
from tensorflow.keras.layers import Dense

from attention import *

class MultiHeadAttention(tf.keras.layers.layer):
    def __init__(self, d_feature, num_heads = 6):
        """
        Args:
            d_model (int): dimentionality of the feature embedding
            num_heads (int): the number of heads
        """
        super(MultiHeadAttention, self).__init__()

        assert d_feature % num_heads == 0

        self.d_feature = d_feature
        self.num_heads = num_heads
        self.q_linear = Dense(d_feature)
        self.k_linear = Dense(d_feature)
        self.v_linear = Dense(d_feature)
        self.out_linear = Dense(d_feature)


    def split_head(self, x):
        """
        Split the last dimension into (num_heads, d_head).
        
        Args:
            x (tensor): input with shape (..., seqlen, d_feature)
        Returns:
            tensor: split tensor with shape (..., num_heads, seqlen, d_head)
        """

        d_head = self.d_feature // self.num_heads
        seqlen = x.shape[1]

        x = tf.reshape(x, (-1, seqlen, self.num_heads, self.d_head))

        return tf.transpose(x, perm=[0, 2, 1, 3])


    def scaled_dot_product_Attention(q, k, v, mask=None):
        """ 
        Scaled dot-product attention.

        Args:
            q (tensor): query with shape (..., q_length, d_feature)
            k (tensor): key with shape (..., k_length, d_feature)
            v (tensor): value with shape (..., v_length, d_feature)
            k_length = v_length
        Returns:
            attention (tensor): self attention with shape (..., q_length, k_length)
        """
        
        assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Embedding dimensions of q, k, v aren't all the same"
        assert k.shape[-2] == v.shape[-2], "Key and value lengths aren't the same"

        depth = q.shape[-1]
        depth = tf.cast(depth, tf.float32)

        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(depth)  # shape (..., q_length, k_length)

        if mask:
            attention_scores += (mask * -1e30)

        attention = tf.nn.softmax(attention_scores, axis=-1)
        attention = tf.matmul(attention, v)  # shape (..., q_length, d_v)

        return attention


    def call(self, q, k, v, mask=None):
        """
        Multi-head Attention.

        Args:
            q (tensor): query with shape (..., q_length, d_feature)
            k (tensor): key with shape (..., k_length, d_feature)
            v (tensor): value with shape (..., v_length, d_feature)
            k_length = v_length
        Returns:
            attention_matrix (tensor): self attention with shape (..., q_length, k_length)
        """

        batch_size = q.shape[0]
        q_length = q.shape[1]

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q_h = self.split_head(q)  # shape (..., num_heads, q_length, d_head)
        k_h = self.split_head(k)  # shape (..., num_heads, k_length, d_head)
        v_h = self.split_head(v)  # shape (..., num_heads, v_length, d_head)

        attention_matrix = self.scaled_dot_product_attention(q_h, k_h, v_h, mask)  # shape (..., num_heads, q_length, d_head)

        attention_matrix = tf.transpose(attention_matrix, perm=[0, 2, 1, 3])  # shape (..., q_length, num_heads, d_head)
        attention_matrix = tf.reshape(attention_matrix, (batch_size, q_length, -1))  # shape (..., q_length, d_feature)

        attention_matrix = self.out_linear(attention_matrix)

        return attention_matrix
        