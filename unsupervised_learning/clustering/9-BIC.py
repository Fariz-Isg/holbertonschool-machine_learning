#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    likelihoods = []
    bics = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        results.append((pi, m, S))
        likelihoods.append(log_l)
        # p = k*1 (pi) + k*d (means) + k*d*(d+1)/2 (cov, symmetric) - 1 (pi sums to 1)
        p = k * (1 + d + d * (d + 1) // 2) - 1
        bics.append(p * np.log(n) - 2 * log_l)

    l = np.array(likelihoods)
    b = np.array(bics)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
