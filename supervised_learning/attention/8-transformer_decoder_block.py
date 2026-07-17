#!/usr/bin/env python3
"""Transformer decoder block implementation."""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Single transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the decoder block.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Hidden units in the feed-forward network.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Run the decoder block forward pass.

        Args:
            x (tf.Tensor): Target input of shape (batch, target_seq_len, dm).
            encoder_output (tf.Tensor): Encoder output of shape
                (batch, input_seq_len, dm).
            training (bool): Whether the model is in training mode.
            look_ahead_mask (tf.Tensor): Mask for the first attention layer.
            padding_mask (tf.Tensor): Mask for the second attention layer.

        Returns:
            tf.Tensor: Output tensor of shape (batch, target_seq_len, dm).
        """
        Q = K = V = x

        output_att1, weights_att1 = self.mha1(
            Q, K, V, mask=look_ahead_mask
        )

        x_drop1 = self.dropout1(output_att1, training=training)
        x = x + x_drop1
        x_norm1 = self.layernorm1(x)

        output_att2, weights_att2 = self.mha2(
            x_norm1,
            encoder_output,
            encoder_output,
            mask=padding_mask
        )
        x_drop2 = self.dropout2(output_att2, training=training)
        x = x_norm1 + x_drop2
        x_norm2 = self.layernorm2(x)

        hidden = self.dense_hidden(x_norm2)
        out = self.dense_output(hidden)
        x_drop3 = self.dropout3(out, training=training)
        x = x_norm2 + x_drop3
        output = self.layernorm3(x)

        return output
