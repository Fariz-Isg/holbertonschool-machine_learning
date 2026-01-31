#!/usr/bin/env python3
"""
Module for cat_matrices2D function
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Args:
        mat1: first matrix
        mat2: second matrix
        axis: axis to concatenate along (0 for rows, 1 for columns)
    Returns:
        New matrix with concatenated matrices, or None if failure
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]
    return None
