#!/usr/bin/env python3
"""Pooling Back Prop module"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer"""
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for h in range(h_new):
        for w in range(w_new):
            if mode == 'max':
                slice_A = A_prev[:,
                                 h * sh: h * sh + kh,
                                 w * sw: w * sw + kw,
                                 :]
                mask = (slice_A == np.max(slice_A, axis=(1, 2),
                                          keepdims=True))
                da = dA[:, h, w, :][:, None, None, :]
                dA_prev[:,
                        h * sh: h * sh + kh,
                        w * sw: w * sw + kw,
                        :] += mask * da
            elif mode == 'avg':
                da = dA[:, h, w, :][:, None, None, :]
                dA_prev[:,
                        h * sh: h * sh + kh,
                        w * sw: w * sw + kw,
                        :] += da / (kh * kw)

    return dA_prev
