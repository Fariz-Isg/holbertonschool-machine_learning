#!/usr/bin/env python3
"""GRU cell implementation."""
import numpy as np


class GRUCell:
    """Represents a GRU cell."""

    def __init__(self, i, h, o):
        """Initialize the GRU cell."""
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward propagation for a single step."""
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        z = self.sigmoid(x_concat @ self.Wz + self.bz)
        r = self.sigmoid(x_concat @ self.Wr + self.br)
        r_h_prev = r * h_prev
        cand_concat = np.concatenate((r_h_prev, x_t), axis=1)
        h_cand = np.tanh(cand_concat @ self.Wh + self.bh)
        h_next = (1 - z) * h_prev + z * h_cand
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y

    @staticmethod
    def softmax(x, axis=-1):
        """Softmax activation function."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))