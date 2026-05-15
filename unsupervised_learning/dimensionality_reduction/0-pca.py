#!/usr/bin/env python3
"""PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) with zero mean across all points
        var: fraction of variance to maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) - the weights matrix
    """
    _, s, vh = np.linalg.svd(X)

    cumulative_var = np.cumsum(s ** 2) / np.sum(s ** 2)
    nd = np.argmax(cumulative_var >= var) + 1

    W = vh[:nd].T

    return W
