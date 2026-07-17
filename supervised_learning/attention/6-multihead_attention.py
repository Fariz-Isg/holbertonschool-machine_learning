#!/usr/bin/env python3
"""Multi-head attention layer implementation."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head attention."""

    def __init__(self, dm, h):
        """Initialize the multi-head attention layer.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=self.dm)
        self.Wk = tf.keras.layers.Dense(units=self.dm)
        self.Wv = tf.keras.layers.Dense(units=self.dm)
        self.linear = tf.keras.layers.Dense(units=self.dm)

    def call(self, Q, K, V, mask):
        """Run the multi-head attention forward pass.

        Args:
            Q (tf.Tensor): Queries of shape (batch, seq_len_q, dk).
            K (tf.Tensor): Keys of shape (batch, seq_len_v, dk).
            V (tf.Tensor): Values of shape (batch, seq_len_v, dv).
            mask (tf.Tensor | None): Optional mask broadcastable to
                (..., seq_len_q, seq_len_v).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Attention output of shape
            (..., seq_len_q, dm) and attention weights of shape
            (..., h, seq_len_q, seq_len_v).
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = tf.reshape(Q, (batch_size, -1, self.h, self.depth))
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.reshape(K, (batch_size, -1, self.h, self.depth))
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.reshape(V, (batch_size, -1, self.h, self.depth))
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        scaled_att, weights_att = sdp_attention(Q, K, V, mask)

        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_att, (batch_size, -1, self.dm))

        output = self.linear(concat_att)

        return output, weights_att
