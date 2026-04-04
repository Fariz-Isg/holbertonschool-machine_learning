#!/usr/bin/env python3
"""Convolutional Back Prop module"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = pw = 0

    A_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    dA_padded = np.zeros(A_padded.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for h in range(h_new):
        for w in range(w_new):
            for c in range(c_new):
                slice_A = A_padded[:,
                                   h * sh: h * sh + kh,
                                   w * sw: w * sw + kw,
                                   :]
                dz = dZ[:, h, w, c]
                dW[:, :, :, c] += np.sum(slice_A * dz[:, None, None, None],
                                         axis=0)
                dA_padded[:,
                          h * sh: h * sh + kh,
                          w * sw: w * sw + kw,
                          :] += W[:, :, :, c] * dz[:, None, None, None]

    if padding == 'same':
        dA = dA_padded[:, ph:h_prev + ph, pw:w_prev + pw, :]
    else:
        dA = dA_padded

    return dA, dW, db
