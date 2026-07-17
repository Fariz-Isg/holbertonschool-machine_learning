#!/usr/bin/env python3
"""Transformer encoder block implementation."""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Single transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the encoder block.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Hidden units in the feed-forward network.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Run the encoder block forward pass.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, dm).
            training (bool): Whether the model is in training mode.
            mask (tf.Tensor | None): Attention mask.

        Returns:
            tf.Tensor: Output tensor of shape (batch, seq_len, dm).
        """
        Q = K = V = x

        output_att, weights_att = self.mha(Q, K, V, mask=mask)

        x_drop1 = self.dropout1(output_att, training=training)
        x = x + x_drop1
        x_norm1 = self.layernorm1(x)

        x_hidden = self.dense_hidden(x_norm1)
        x_output = self.dense_output(x_hidden)

        x_drop2 = self.dropout2(x_output, training=training)
        x = x_norm1 + x_drop2
        output = self.layernorm2(x)

        return output
