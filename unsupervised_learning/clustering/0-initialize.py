#!/usr/bin/env python3
"""
Module to initialize cluster centroids for K-means.
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    try:
        low = np.min(X, axis=0)
        high = np.max(X, axis=0)
        return np.random.uniform(low, high, size=(k, X.shape[1]))
    except Exception:
        return None
