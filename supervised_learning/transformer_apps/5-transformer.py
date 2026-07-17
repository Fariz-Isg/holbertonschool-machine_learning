#!/usr/bin/env python3
"""Self-contained transformer model for machine translation.

Inlines all sub-components (positional encoding, scaled dot-product
attention, multi-head attention, encoder/decoder blocks) so this file
can be imported standalone without the numbered attention modules.
"""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Compute sinusoidal positional encodings.

    Args:
        max_seq_len (int): Maximum sequence length.
        dm (int): Model dimensionality.

    Returns:
        np.ndarray: Shape (max_seq_len, dm).
    """
    pe = np.zeros((max_seq_len, dm))
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dims = np.arange(dm // 2)[np.newaxis, :]
    denominators = 10000 ** ((2 * dims) / dm)
    pe[:, 0::2] = np.sin(positions / denominators)
    pe[:, 1::2] = np.cos(positions / denominators)
    return pe


def sdp_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Args:
        Q: Queries (..., seq_len_q, dk).
        K: Keys (..., seq_len_v, dk).
        V: Values (..., seq_len_v, dv).
        mask: Optional broadcastable mask.

    Returns:
        (output, weights) tensors.
    """
    dot = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = dot / tf.math.sqrt(dk)
    if mask is not None:
        scaled += mask * -1e9
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer."""

    def __init__(self, dm, h):
        """Initialize multi-head attention.

        Args:
            dm (int): Model dimensionality.
            h (int): Number of heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """Forward pass.

        Args:
            Q: Queries (batch, seq_q, dm).
            K: Keys (batch, seq_v, dm).
            V: Values (batch, seq_v, dm).
            mask: Optional attention mask.

        Returns:
            (output, weights) tensors.
        """
        batch = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = tf.transpose(
            tf.reshape(Q, (batch, -1, self.h, self.depth)),
            perm=[0, 2, 1, 3])
        K = tf.transpose(
            tf.reshape(K, (batch, -1, self.h, self.depth)),
            perm=[0, 2, 1, 3])
        V = tf.transpose(
            tf.reshape(V, (batch, -1, self.h, self.depth)),
            perm=[0, 2, 1, 3])

        scaled, weights = sdp_attention(Q, K, V, mask)
        scaled = tf.transpose(scaled, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled, (batch, -1, self.dm))
        return self.linear(concat), weights


class EncoderBlock(tf.keras.layers.Layer):
    """Single transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize encoder block.

        Args:
            dm (int): Model dimensionality.
            h (int): Number of heads.
            hidden (int): Feed-forward hidden units.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Forward pass.

        Args:
            x: Input (batch, seq_len, dm).
            training: Boolean training flag.
            mask: Optional attention mask.

        Returns:
            Output tensor (batch, seq_len, dm).
        """
        attn_out, _ = self.mha(x, x, x, mask=mask)
        x = self.layernorm1(x + self.dropout1(attn_out, training=training))
        ff_out = self.dense_output(self.dense_hidden(x))
        return self.layernorm2(x + self.dropout2(ff_out, training=training))


class DecoderBlock(tf.keras.layers.Layer):
    """Single transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize decoder block.

        Args:
            dm (int): Model dimensionality.
            h (int): Number of heads.
            hidden (int): Feed-forward hidden units.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        """Forward pass.

        Args:
            x: Target input (batch, target_seq_len, dm).
            enc_out: Encoder output (batch, input_seq_len, dm).
            training: Boolean training flag.
            look_ahead_mask: Mask for masked self-attention.
            padding_mask: Mask for cross-attention.

        Returns:
            Output tensor (batch, target_seq_len, dm).
        """
        attn1, _ = self.mha1(x, x, x, mask=look_ahead_mask)
        x = self.layernorm1(x + self.dropout1(attn1, training=training))
        attn2, _ = self.mha2(x, enc_out, enc_out, mask=padding_mask)
        x = self.layernorm2(x + self.dropout2(attn2, training=training))
        ff = self.dense_output(self.dense_hidden(x))
        return self.layernorm3(x + self.dropout3(ff, training=training))


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder stack."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initialize encoder.

        Args:
            N (int): Number of encoder blocks.
            dm (int): Model dimensionality.
            h (int): Number of heads.
            hidden (int): Feed-forward hidden units.
            input_vocab (int): Input vocabulary size.
            max_seq_len (int): Maximum sequence length.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Forward pass.

        Args:
            x: Input tokens (batch, input_seq_len).
            training: Boolean training flag.
            mask: Padding mask.

        Returns:
            Encoder output (batch, input_seq_len, dm).
        """
        x = self.embedding(x)
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """Transformer decoder stack."""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initialize decoder.

        Args:
            N (int): Number of decoder blocks.
            dm (int): Model dimensionality.
            h (int): Number of heads.
            hidden (int): Feed-forward hidden units.
            target_vocab (int): Target vocabulary size.
            max_seq_len (int): Maximum sequence length.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        """Forward pass.

        Args:
            x: Target tokens (batch, target_seq_len).
            enc_out: Encoder output (batch, input_seq_len, dm).
            training: Boolean training flag.
            look_ahead_mask: Look-ahead mask.
            padding_mask: Padding mask.

        Returns:
            Decoder output (batch, target_seq_len, dm).
        """
        x = self.embedding(x)
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, enc_out, training, look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    """Complete encoder-decoder transformer network."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialize the transformer.

        Args:
            N (int): Number of encoder/decoder blocks.
            dm (int): Model dimensionality.
            h (int): Number of heads.
            hidden (int): Feed-forward hidden units.
            input_vocab (int): Input vocabulary size.
            target_vocab (int): Target vocabulary size.
            max_seq_input (int): Max input sequence length.
            max_seq_target (int): Max target sequence length.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Forward pass.

        Args:
            inputs: Input tokens (batch, input_seq_len).
            target: Target tokens (batch, target_seq_len).
            training: Boolean training flag.
            encoder_mask: Encoder padding mask.
            look_ahead_mask: Decoder look-ahead mask.
            decoder_mask: Decoder cross-attention padding mask.

        Returns:
            Logits of shape (batch, target_seq_len, target_vocab).
        """
        enc = self.encoder(inputs, training, encoder_mask)
        dec = self.decoder(target, enc, training,
                           look_ahead_mask, decoder_mask)
        return self.linear(dec)
