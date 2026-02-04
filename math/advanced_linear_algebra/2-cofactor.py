#!/usr/bin/env python3
"""
Calculates the cofactor matrix of a matrix
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    Args:
        matrix: list of lists whose cofactor matrix should be calculated
    Returns:
        The cofactor matrix of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cofactors = []
    for r in range(n):
        row_cofactors = []
        for c in range(n):
            sub_matrix = [x[:c] + x[c+1:]
                          for i, x in enumerate(matrix) if i != r]
            det = determinant(sub_matrix)
            row_cofactors.append(((-1) ** (r + c)) * det)
        cofactors.append(row_cofactors)

    return cofactors


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    Args:
        matrix: list of lists
    Returns:
        The determinant of the matrix
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Special case for [[]] which represents 0x0 matrix
    if matrix == [[]]:
        return 1

    # Check if matrix is square
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # 1x1 case
    if n == 1:
        return matrix[0][0]

    # 2x2 case
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive step (Laplace expansion)
    det = 0
    for c in range(n):
        # Create submatrix by removing row 0 and column c
        sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)

    return det
