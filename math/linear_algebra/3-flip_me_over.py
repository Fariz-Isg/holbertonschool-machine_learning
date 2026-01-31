#!/usr/bin/env python3
"""
Module for matrix_transpose function
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix
    Args:
        matrix: the matrix to transpose
    Returns:
        A new matrix that is the transpose of the input matrix
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
