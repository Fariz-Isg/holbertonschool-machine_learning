#!/usr/bin/env python3
"""Same Convolution module"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph_top = (kh - 1) // 2
    ph_bottom = kh // 2
    pw_left = (kw - 1) // 2
    pw_right = kw // 2

    images_padded = np.pad(
        images,
        ((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = images_padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
