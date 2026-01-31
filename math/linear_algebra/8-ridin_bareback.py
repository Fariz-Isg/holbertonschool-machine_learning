#!/usr/bin/env python3
"""
Module for mat_mul function
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication
    Args:
        mat1: first matrix
        mat2: second matrix
    Returns:
        New matrix with product, or None if failure
    """
    if len(mat1[0]) != len(mat2):
        return None

    # Result dimensions: rows of mat1 x cols of mat2
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
