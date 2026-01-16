#!/usr/bin/env python3
"""
Module to calculate summation of squares
"""

def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n
    Args:
        n (int): the stopping condition
    Returns:
        int: the sum of squares, or None if n is invalid
    """
    if type(n) is not int or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
