#!/usr/bin/env python3
"""LSTM cell implementation."""
import numpy as np


def softmax(x):
    """Compute row-wise softmax."""
    max_x = np.amax(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def sigmoid(x):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-x))


class LSTMCell:
    """Represents an LSTM cell."""

    def __init__(self, i, h, o):
        """Initialize LSTM cell."""
        i_h_concat = i + h
        self.Wf = np.random.normal(size=(i_h_concat, h))
        self.Wu = np.random.normal(size=(i_h_concat, h))
        self.Wc = np.random.normal(size=(i_h_concat, h))
        self.Wo = np.random.normal(size=(i_h_concat, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros(shape=(1, h))
        self.bu = np.zeros(shape=(1, h))
        self.bc = np.zeros(shape=(1, h))
        self.bo = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, c_prev, x_t):
        """One LSTM step forward."""
        hx = np.concatenate((h_prev, x_t), axis=1)
        f_t = sigmoid(hx @ self.Wf + self.bf)
        u_t = sigmoid(hx @ self.Wu + self.bu)
        c_bar = np.tanh(hx @ self.Wc + self.bc)
        o_t = sigmoid(hx @ self.Wo + self.bo)

        c_next = f_t * c_prev + u_t * c_bar
        h_next = o_t * np.tanh(c_next)

        y = softmax(h_next @ self.Wy + self.by)

        return h_next, c_next, y