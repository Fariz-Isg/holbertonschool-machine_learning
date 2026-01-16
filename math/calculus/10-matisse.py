#!/usr/bin/env python3
"""
Module to calculate the derivative of a polynomial
"""

def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial
    Args:
        poly (list): a list of coefficients representing a polynomial
                     the index of the list represents the power of x
    Returns:
        list: a new list of coefficients representing the derivative
              of the polynomial, or None if poly is invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]

    if not derivative:
        return [0]

    return derivative
