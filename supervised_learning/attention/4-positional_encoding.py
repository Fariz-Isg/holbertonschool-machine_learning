#!/usr/bin/env python3
"""Positional encoding utility for Transformers."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Compute positional encodings for a transformer model.

    Args:
        max_seq_len (int): Maximum sequence length.
        dm (int): Model depth.

    Returns:
        np.ndarray: Positional encodings of shape (max_seq_len, dm).
    """
    pe_vector = np.zeros(shape=(max_seq_len, dm))
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dims = np.arange(int(dm / 2))[np.newaxis, :]
    denominators = 10000 ** ((2 * dims) / dm)

    pe_vector[:, 0::2] = np.sin(positions / denominators)
    pe_vector[:, 1::2] = np.cos(positions / denominators)

    return pe_vector
