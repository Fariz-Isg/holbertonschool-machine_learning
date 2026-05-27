#!/usr/bin/env python3
"""
Module to initialize variables for a Gaussian Mixture Model.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    try:
        n, d = X.shape
        pi = np.full((k,), 1 / k)
        m, _ = kmeans(X, k)
        if m is None:
            return None, None, None
        S = np.tile(np.eye(d), (k, 1, 1))
        return pi, m, S
    except Exception:
        return None, None, None
