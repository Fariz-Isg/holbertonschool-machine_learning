#!/usr/bin/env python3
"""Self-attention layer for sequence models."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Compute Bahdanau-style attention for sequence-to-sequence models."""

    def __init__(self, units):
        """Initialize the attention layer.

        Args:
            units (int): Number of hidden units in the alignment model.

        Raises:
            TypeError: If `units` is not an integer.
        """
        if not isinstance(units, int):
            raise TypeError("units should be an integer")

        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """Calculate attention context and weights.

        Args:
            s_prev (tf.Tensor): Previous decoder state of shape (batch, units).
            hidden_states (tf.Tensor): Encoder outputs of shape
                (batch, input_seq_len, units).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Context vector of shape
            (batch, units) and attention weights of shape
            (batch, input_seq_len, 1).
        """
        s_prev = tf.expand_dims(s_prev, axis=1)
        scores = self.V(
            tf.nn.tanh(self.W(s_prev) + self.U(hidden_states))
        )

        att_weights = tf.nn.softmax(scores, axis=1)

        context = att_weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, att_weights
