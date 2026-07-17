#!/usr/bin/env python3
"""RNN decoder layer implementation."""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN decoder for sequence-to-sequence translation tasks."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNN decoder.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.

        Raises:
            TypeError: If any provided parameter is not an integer.
        """
        invalid_args = [arg for arg in [vocab, embedding, units, batch]
                        if not isinstance(arg, int)]
        if invalid_args:
            arg_str = ", ".join([f"{arg}" for arg in invalid_args])
            raise TypeError(f"{arg_str} Should be an integer.")

        super().__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """Run the decoder forward pass.

        Args:
            x (tf.Tensor): Previous target token of shape (batch, 1).
            s_prev (tf.Tensor): Previous decoder state of shape
                (batch, units).
            hidden_states (tf.Tensor): Encoder outputs of shape
                (batch, input_seq_len, units).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Logits of shape (batch, vocab) and
            the new decoder state of shape (batch, units).
        """
        context, att_weights = self.attention(s_prev, hidden_states)

        x = self.embedding(x)

        context = tf.expand_dims(context, axis=1)
        context_concat = tf.concat([context, x], axis=-1)

        outputs, hidden_state = self.gru(context_concat)
        new_outputs = tf.reshape(outputs,
                                 shape=(outputs.shape[0], outputs.shape[2]))

        y = self.F(new_outputs)

        return y, hidden_state
