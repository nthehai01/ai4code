import tensorflow as tf

def ScaledDotProductAttention(q, k, v, mask=None):
    """ 
    Scaled dot-product attention.

    Args:
        q (tensor): query with shape (..., q_length, d_k)
        k (tensor): key with shape (..., k_length, d_k)
        v (tensor): value with shape (..., v_length, d_k)
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
