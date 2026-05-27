#!/usr/bin/env python3
"""
Module to calculate intra-cluster variance.
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    try:
        dists = np.sum((X[:, None] - C) ** 2, axis=-1)
        min_dists = np.min(dists, axis=-1)
        return np.sum(min_dists)
    except Exception:
        return None
