#!/usr/bin/env python3
"""Transformer decoder implementation."""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Transformer decoder composed of stacked decoder blocks."""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initialize the transformer decoder.

        Args:
            N (int): Number of decoder blocks.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Hidden units in the feed-forward network.
            target_vocab (int): Size of the target vocabulary.
            max_seq_len (int): Maximum sequence length.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = []
        for i in range(N):
            self.blocks.append(DecoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Run the decoder forward pass.

        Args:
            x (tf.Tensor): Target input of shape
                (batch, target_seq_len).
            encoder_output (tf.Tensor): Encoder output of shape
                (batch, input_seq_len, dm).
            training (bool): Whether the model is in training mode.
            look_ahead_mask (tf.Tensor): Mask for the first
                multi-head attention layer.
            padding_mask (tf.Tensor): Mask for the second
                multi-head attention layer.

        Returns:
            tf.Tensor: Decoder output of shape
            (batch, target_seq_len, dm).
        """
        x = self.embedding(x)
        target_seq_len = tf.shape(x)[1]

        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:target_seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, encoder_output, training,
                      look_ahead_mask, padding_mask)

        return x
