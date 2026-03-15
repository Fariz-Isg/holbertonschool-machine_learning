#!/usr/bin/env python3
"""Momentum module"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent
    with momentum optimization algorithm"""
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new
    return var_new, v_new
