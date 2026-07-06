#!/usr/bin/env python3
'''RNN implementation'''
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    - X is the data to be used, given
    as a numpy.ndarray of shape (t, m, i)
    - h_0 is the initial hidden state,
    given as a numpy.ndarray of shape (m, h)
    """
    # _, m, i = X.shape
    # _, h = h_0.shape
    # cell = rnn_cell(i, h, o)
    H, Y = [h_0], []
    h_current = h_0
    for x_t in X:
        h_current, y = rnn_cell.forward(h_current, x_t)
        H.append(h_current)
        Y.append(y)
    return np.array(H), np.array(Y)