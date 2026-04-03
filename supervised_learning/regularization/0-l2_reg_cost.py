#!/usr/bin/env python3
"""L2 Regularization Cost module"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost with L2 regularization"""
    l2_reg = 0
    for i in range(1, L + 1):
        l2_reg += np.sum(np.square(weights['W' + str(i)]))
    return cost + (lambtha / (2 * m)) * l2_reg
