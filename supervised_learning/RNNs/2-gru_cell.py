#!/usr/bin/env python3
'''GRU implementation'''
import numpy as np


class GRUCell:
    """
    Gated Recurrent Unit
    """
    def __init__(self, i, h, o):
        """
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs
        """
        # update gate
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        # reset gate
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        # intermediate hidden state
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))
        # output
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        '''
        - x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell
        - m is the batche size for the data
        - h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
        '''
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        z = self.sigmoid(x_concat @ self.Wz + self.bz)  # (m, h)
        r = self.sigmoid(x_concat @ self.Wr + self.br)  # (m, h)
        r_h_prev = r * h_prev
        cand_concat = np.concatenate((r_h_prev, x_t), axis=1)  # (m, h + i)
        h_cand = np.tanh(cand_concat @ self.Wh + self.bh)  # (m, h)
        h_next = (1 - z) * h_prev + z * h_cand  # (m, h)
        y = self.softmax(h_next @ self.Wy + self.by)  # (m, o)
        return h_next, y

    @staticmethod
    def softmax(x, axis=-1):
        """Softmax activation function"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """Sigmoid Activation"""
        return 1 / (1 + np.exp(-x))