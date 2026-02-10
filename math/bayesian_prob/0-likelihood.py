#!/usr/bin/env python3
"""
Likelihood function
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # L(P | n, x) = (n! / (x! * (n - x)!)) * P^x * (1 - P)^(n - x)
    # factorials can be large, so we compute the logic slightly differently
    # or just use np.math.factorial? The user prompt implies using numpy.
    # However x and n are scalars, so we can calculate the coefficient once.

    # Binomial coefficient: n! / (x! * (n-x)!)
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)
    comb = fact_n / (fact_x * fact_nx)

    L = comb * (P ** x) * ((1 - P) ** (n - x))
    return L
