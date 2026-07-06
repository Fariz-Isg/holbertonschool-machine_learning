#!/usr/bin/env python3
"""Deep RNN forward propagation."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Forward propagation for a deep RNN.

    Args:
        rnn_cells: List of RNN cell instances for each layer.
        X (np.ndarray): Input data of shape (t, m, i).
        h_0 (np.ndarray): Initial hidden states of shape (l, m, h).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - H: Hidden states, shape (t + 1, l, m, h), with H[0] = h_0.
            - Y: Outputs for each time step, shape (t, m, o).

    Raises:
        TypeError: If X or h_0 are not numpy arrays.
        ValueError: If X and h_0 have incompatible shapes.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('X should be a ndarray')
    if not isinstance(h_0, np.ndarray):
        raise TypeError('h_0 should be a ndarray')
    if X.shape[1] != h_0.shape[1]:
        raise ValueError("X.shape is (t,m,i) and h_0 shape is (l,m,h)")

    t, m, _ = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y_steps = []

    for t_idx in range(t):
        h_prev_layer = None
        for layer in range(l):
            x_in = X[t_idx] if layer == 0 else h_prev_layer
            h_curr, y_curr = rnn_cells[layer].forward(H[t_idx, layer], x_in)
            H[t_idx + 1, layer] = h_curr
            h_prev_layer = h_curr
        Y_steps.append(y_curr)

    Y = np.array(Y_steps)
    return H, Y