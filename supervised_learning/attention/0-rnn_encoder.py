#!/usr/bin/env python3
"""RNN encoder layer implementation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN encoder for sequence-to-sequence translation tasks."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNN encoder.

        Args:
            vocab (int): Size of the input vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.

        Raises:
            TypeError: If any provided parameter is not an integer.
        """
        if not isinstance(vocab, int):
            raise TypeError("vocab Should be an integer")
        if not isinstance(embedding, int):
            raise TypeError("embedding Should be an integer")
        if not isinstance(units, int):
            raise TypeError("units Should be an integer")
        if not isinstance(batch, int):
            raise TypeError("batch Should be an integer")

        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding,
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer=tf.keras.initializers.GlorotUniform(),
        )

    def initialize_hidden_state(self):
        """Return a zero-initialized hidden state tensor."""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """Run the encoder forward pass.

        Args:
            x (tf.Tensor): Token indices of shape (batch, input_seq_len).
            initial (tf.Tensor): Initial hidden state
                with shape (batch, units).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Tuple of:
                outputs of shape (batch, input_seq_len, units) and the last
                hidden state of shape (batch, units).
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
