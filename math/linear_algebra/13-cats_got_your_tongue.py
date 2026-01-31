#!/usr/bin/env python3
"""
Module for np_cat function
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Args:
        mat1: first matrix
        mat2: second matrix
        axis: axis to concatenate along
    Returns:
        new numpy.ndarray
    """
    return np.concatenate((mat1, mat2), axis=axis)
