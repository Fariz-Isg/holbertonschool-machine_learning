#!/usr/bin/env python3
"""Convolutional Forward Prop module"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph = pw = 0

    A_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    output_h = int((h_prev + 2 * ph - kh) / sh) + 1
    output_w = int((w_prev + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                image_slice = A_padded[:,
                                       i * sh: i * sh + kh,
                                       j * sw: j * sw + kw,
                                       :]
                kernel = W[:, :, :, k]
                output[:, i, j, k] = np.sum(image_slice * kernel,
                                            axis=(1, 2, 3))

    Z = output + b
    return activation(Z)
