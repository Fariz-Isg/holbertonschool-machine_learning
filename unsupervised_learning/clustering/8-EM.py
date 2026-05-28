#!/usr/bin/env python3
"""Expectation Maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters
        iterations: positive integer, maximum number of iterations
        tol: non-negative float, tolerance for log likelihood
        verbose: boolean, whether to print log likelihood info

    Returns:
        pi, m, S, g, l or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    l_prev = 0
    for i in range(iterations):
        g, ll = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(ll, 5)))
        if i > 0 and abs(ll - l_prev) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(ll, 5)))
            return pi, m, S, g, ll
        pi, m, S = maximization(X, g)
        l_prev = ll
    g, ll = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            iterations, round(ll, 5)))
    return pi, m, S, g, ll
