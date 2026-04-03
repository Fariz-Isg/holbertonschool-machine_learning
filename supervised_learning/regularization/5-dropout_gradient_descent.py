#!/usr/bin/env python3
"""Gradient Descent with Dropout module"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights and biases with Dropout"""
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dw = np.dot(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            D = cache['D' + str(i - 1)]
            dA = np.dot(W.T, dz)
            dA = (dA * D) / keep_prob
            # A_prev is after dropout scaling: A = tanh(Z) * D / keep_prob
            # So tanh(Z) = A_prev * keep_prob (where D=1)
            dz = dA * (1 - (A_prev * keep_prob)**2)

        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = b - alpha * db
