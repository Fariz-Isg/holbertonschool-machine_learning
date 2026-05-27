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

        # Update centroids
        for i in range(k):
            points = X[clss == i]
            if len(points) == 0:
                C[i] = np.random.uniform(low, high)
            else:
                C[i] = np.mean(points, axis=0)

        if np.all(C == C_prev):
            break

    # Recalculate clss one final time for the final centroids
    dists = np.sum((X[:, None] - C) ** 2, axis=-1)
    clss = np.argmin(dists, axis=-1)

    return C, clss
