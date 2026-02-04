#!/usr/bin/env python3
"""
Calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    Args:
        matrix: numpy.ndarray of shape (n, n)
    Returns:
        The string describing the definiteness or None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check validity: square and symmetric
    if matrix.ndim != 2:
        return None
    r, c = matrix.shape
    if r != c or r == 0:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        w, _ = np.linalg.eig(matrix)
    except Exception:
        return None

    if np.all(w > 0):
        return "Positive definite"
    if np.all(w >= 0):
        return "Positive semi-definite"
    if np.all(w < 0):
        return "Negative definite"
    if np.all(w <= 0):
        return "Negative semi-definite"

    return "Indefinite"
