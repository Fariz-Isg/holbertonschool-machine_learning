#!/usr/bin/env python3
"""RNN cell implementation."""
import numpy as np


class RNNCell:
    """Represents an RNN cell."""

    def __init__(self, i, h, o):
        """Initialize the RNN cell."""
        self.Wh, self.Wy = (np.random.randn(i + h, h),
                            np.random.randn(h, o))
        self.bh, self.by = (np.zeros((1, h)),
                            np.zeros((1, o)))

    def forward(self, h_prev, x_t):
        """Forward propagation for a single step."""
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(x_concat @ self.Wh + self.bh)
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y

    @staticmethod
    def softmax(x, axis=-1):
        """Softmax activation function."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)