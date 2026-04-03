#!/usr/bin/env python3
"""Valid Convolution module"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w))

    # Perform convolution using exactly two loops
    for i in range(output_h):
        for j in range(output_w):
            image_slice = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
