#!/usr/bin/env python3
"""
Module for matrix_shape function
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix
    Args:
        matrix: the matrix to calculate the shape of
    Returns:
        list of integers representing the shape
    """
    if not isinstance(matrix, list):
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
