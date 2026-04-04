#!/usr/bin/env python3
"""Multiple Kernels Convolution module"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
        ph_top = ph
        ph_bottom = ph
        pw_left = pw
        pw_right = pw
    elif padding == 'valid':
        ph_top = ph_bottom = 0
        pw_left = pw_right = 0
    else:
        ph_top = ph_bottom = padding[0]
        pw_left = pw_right = padding[1]

    images_padded = np.pad(
        images,
        ((0, 0), (ph_top, ph_bottom), (pw_left, pw_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    output_h = int((h + ph_top + ph_bottom - kh) / sh) + 1
    output_w = int((w + pw_left + pw_right - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                image_slice = images_padded[:,
                                            i * sh: i * sh + kh,
                                            j * sw: j * sw + kw,
                                            :]
                kernel = kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(image_slice * kernel,
                                            axis=(1, 2, 3))

    return output
