#!/usr/bin/env python3
"""Pooling module"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int((h - kh) / sh) + 1
    output_w = int((w - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            image_slice = images[:,
                                 i * sh: i * sh + kh,
                                 j * sw: j * sw + kw,
                                 :]
            if mode == 'max':
                output[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return output
