#!/usr/bin/env python3
"""
Module to calculate the maximization step in the EM algorithm for a GMM.
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None
    try:
        n_k = np.sum(g, axis=1)
        pi = n_k / n
        m = (g @ X) / n_k[:, None]
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            S[i] = (g[i] * diff.T) @ diff / n_k[i]
        return pi, m, S
    except Exception:
        return None, None, None
