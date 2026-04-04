#!/usr/bin/env python3
"""Pooling Forward Prop module"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int((h_prev - kh) / sh) + 1
    output_w = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, output_h, output_w, c_prev))

    for i in range(output_h):
        for j in range(output_w):
            slice_A = A_prev[:,
                             i * sh: i * sh + kh,
                             j * sw: j * sw + kw,
                             :]
            if mode == 'max':
                A[:, i, j, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(slice_A, axis=(1, 2))

    return A
