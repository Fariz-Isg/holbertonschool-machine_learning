#!/usr/bin/env python3
"""One-hot decoding module"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels
    Args:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
    Returns:
        a numpy.ndarray with shape (m, ) containing the numeric labels
        for each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
