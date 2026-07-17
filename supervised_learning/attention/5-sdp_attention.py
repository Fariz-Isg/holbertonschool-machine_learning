#!/usr/bin/env python3
"""Scaled dot-product attention implementation."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot-product attention.

    Args:
        Q (tf.Tensor): Query tensor of shape (..., seq_len_q, dk).
        K (tf.Tensor): Key tensor of shape (..., seq_len_v, dk).
        V (tf.Tensor): Value tensor of shape (..., seq_len_v, dv).
        mask (tf.Tensor | None): Optional tensor broadcastable to
            (..., seq_len_q, seq_len_v).

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Attention output of shape
        (..., seq_len_q, dv) and attention weights of shape
        (..., seq_len_q, seq_len_v).
    """
    dot_product = tf.matmul(Q, K, transpose_b=True)

    d_k = tf.cast(tf.shape(K)[-1], tf.float32)

    scaling = dot_product / tf.math.sqrt(d_k)

    if mask is not None:
        scaling += mask * -1e9

    attention_weight = tf.nn.softmax(scaling, axis=-1)

    output = tf.matmul(attention_weight, V)

    return output, attention_weight
