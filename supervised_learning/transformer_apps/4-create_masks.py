#!/usr/bin/env python3
"""Mask creation utilities for transformer training."""
import tensorflow as tf


def create_masks(inputs, target):
    """Create all masks needed for transformer training.

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) with
            input token IDs.
        target: tf.Tensor of shape (batch_size, seq_len_out) with
            target token IDs.

    Returns:
        Tuple of (encoder_mask, combined_mask, decoder_mask):
            encoder_mask: shape (batch, 1, 1, seq_len_in) — padding
                mask for encoder self-attention.
            combined_mask: shape (batch, 1, seq_len_out, seq_len_out)
                — look-ahead + padding mask for decoder
                self-attention.
            decoder_mask: shape (batch, 1, 1, seq_len_in) — padding
                mask for decoder cross-attention on encoder output.
    """
    # Encoder padding mask: 1.0 where input is 0 (padded)
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask (for cross-attention on encoder output)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Look-ahead mask: upper-triangular matrix of 1s
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
    )

    # Target padding mask
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Combined mask: maximum of look-ahead and target padding
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
