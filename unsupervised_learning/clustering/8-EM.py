#!/usr/bin/env python3
"""
Module to perform the expectation
 maximization algorithm for a GMM.
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5,
                             verbose=False):
    """
    Performs the expectation maximization for a GMM.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if type(k) is not int or k <= 0:
        return None, None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None, None

    try:
        pi, m, S = initialize(X, k)

        if pi is None:
            return None, None, None, None, None

        g, log_likelihood = expectation(X, pi, m, S)

        if g is None:
            return None, None, None, None, None

        if verbose:
            print('Log Likelihood after 0 iterations: {}'.format(
                round(log_likelihood, 5)))

        for i in range(1, iterations + 1):
            pi, m, S = maximization(X, g)

            if pi is None:
                return None, None, None, None, None

            g, new_log_likelihood = expectation(X, pi, m, S)

            if g is None:
                return None, None, None, None, None

            if abs(new_log_likelihood - log_likelihood) <= tol:
                log_likelihood = new_log_likelihood

                if verbose:
                    print('Log Likelihood after {} iterations: {}'.format(
                        i, round(log_likelihood, 5)))
                break

            log_likelihood = new_log_likelihood

            if verbose and i % 10 == 0:
                print('Log Likelihood after {} iterations: {}'.format(
                    i, round(log_likelihood, 5)))

        return pi, m, S, g, log_likelihood

    except Exception:
        return None, None, None, None, None
