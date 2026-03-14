#!/usr/bin/env python3
"""One-hot encoding module"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels
        classes: the maximum number of classes found in Y
    Returns:
        a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None

    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
