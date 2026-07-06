#!/usr/bin/env python3
"""RNN forward propagation."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN."""
    H, Y = [h_0], []
    h_current = h_0
    for x_t in X:
        h_current, y = rnn_cell.forward(h_current, x_t)
        H.append(h_current)
        Y.append(y)
    return np.array(H), np.array(Y)