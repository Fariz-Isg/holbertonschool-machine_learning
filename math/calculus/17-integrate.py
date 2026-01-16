#!/usr/bin/env python3
"""
Module to calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.
    Args:
        poly (list): a list of coefficients representing a polynomial
                     the index of the list represents the power of x
        C (int): an integer representing the integration constant
    Returns:
        list: a new list of coefficients representing the integral
              of the polynomial, or None if poly or C are invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    if poly == [0]:
        return [C]

    integral = [C]
    for i in range(len(poly)):
        exponent = i + 1
        coefficient = poly[i] / exponent
        if coefficient.is_integer():
            coefficient = int(coefficient)
        integral.append(coefficient)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    
    return integral
