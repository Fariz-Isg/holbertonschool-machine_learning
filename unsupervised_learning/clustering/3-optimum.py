#!/usr/bin/env python3
"""
Module to find the optimum number of clusters.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    try:
        results = []
        variances = []
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            if C is None or clss is None:
                return None, None
            var = variance(X, C)
            if var is None:
                return None, None
            results.append((C, clss))
            variances.append(var)
        d_vars = [variances[0] - var for var in variances]
        return results, d_vars
    except Exception:
        return None, None
