#!/usr/bin/env python3
"""Transformer encoder implementation."""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder composed of stacked encoder blocks."""
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initialize the transformer encoder.

        Args:
            N (int): Number of encoder blocks.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Hidden units in the feed-forward network.
            input_vocab (int): Size of the input vocabulary.
            max_seq_len (int): Maximum sequence length.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = []
        for i in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Run the encoder forward pass.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len).
            training (bool): Whether the model is in training mode.
            mask (tf.Tensor | None): Attention mask.

        Returns:
            tf.Tensor: Encoder output of shape
            (batch, input_seq_len, dm).
        """
        x = self.embedding(x)
        input_seq_len = tf.shape(x)[1]

        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x
