#!/usr/bin/env python3
"""
Module to perform K-means clustering.
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    n, d = X.shape
    if n == 0 or d == 0:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        C_prev = C.copy()

        # Distance calculation: (n, k)
        dists = np.sum((X[:, None] - C) ** 2, axis=-1)
        clss = np.argmin(dists, axis=-1)

        # Update centroids: Vectorized without nested loops
        sum_X = np.zeros((k, d))
        np.add.at(sum_X, clss, X)
        counts = np.bincount(clss, minlength=k)[:, None]
        safe_counts = np.where(counts == 0, 1, counts)
        C_new = sum_X / safe_counts

        empty_mask = (counts[:, 0] == 0)
        num_empty = np.sum(empty_mask)
        if num_empty > 0:
            C_new[empty_mask] = np.random.uniform(
                low, high, size=(num_empty, d)
            )

        C = C_new
        if np.all(C == C_prev):
            break

    # Recalculate clss one final time for the final centroids
    dists = np.sum((X[:, None] - C) ** 2, axis=-1)
    clss = np.argmin(dists, axis=-1)

    return C, clss
