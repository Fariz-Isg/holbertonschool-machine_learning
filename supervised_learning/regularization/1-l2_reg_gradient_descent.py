#!/usr/bin/env python3
"""L2 Regularization Gradient Descent module"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases with L2 regularization"""
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dw = (np.matmul(dz, A_prev.T) / m) + (lambtha / m * W)
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            dz = np.matmul(W.T, dz) * (1 - (A_prev ** 2))

        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = b - alpha * db
