#!/usr/bin/env python3
"""
Module for np_elementwise function
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    Args:
        mat1: first matrix/scalar
        mat2: second matrix/scalar
    Returns:
        tuple containing (sum, difference, product, quotient)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
