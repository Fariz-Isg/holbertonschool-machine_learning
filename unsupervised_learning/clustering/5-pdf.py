#!/usr/bin/env python3
"""
Module to calculate the PDF of a Gaussian distribution.
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    d = X.shape[1]
    if m.shape[0] != d or S.shape != (d, d):
        return None
    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
        diff = X - m
        coeff = 1 / (np.sqrt((2 * np.pi) ** d * det))
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        P = coeff * np.exp(exponent)
        P = np.maximum(P, 1e-300)
        return P
    except Exception:
        return None
