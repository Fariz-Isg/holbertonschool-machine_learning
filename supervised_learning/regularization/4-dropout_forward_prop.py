#!/usr/bin/env python3
"""Forward Propagation with Dropout module"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagation with inverted dropout"""
    cache = {'A0': X}
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.dot(W, A_prev) + b

        if i == L:
            # Softmax
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache['A' + str(i)] = A
        else:
            # Tanh + Dropout
            A = np.tanh(Z)
            D = (np.random.rand(*(A.shape)) < keep_prob).astype(int)
            A = (A * D) / keep_prob
            cache['A' + str(i)] = A
            cache['D' + str(i)] = D

    return cache
