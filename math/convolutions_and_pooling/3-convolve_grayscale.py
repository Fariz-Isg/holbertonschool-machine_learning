#!/usr/bin/env python3
"""Strided Convolution module"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        out_h = int(np.ceil(h / sh))
        pad_tot_h = max((out_h - 1) * sh + kh - h, 0)
        ph_top = pad_tot_h // 2 + pad_tot_h % 2
        ph_bottom = pad_tot_h // 2

        out_w = int(np.ceil(w / sw))
        pad_tot_w = max((out_w - 1) * sw + kw - w, 0)
        pw_left = pad_tot_w // 2 + pad_tot_w % 2
        pw_right = pad_tot_w // 2
    elif padding == 'valid':
        ph_top = ph_bottom = 0
        pw_left = pw_right = 0
    else:
        ph_top = ph_bottom = padding[0]
        pw_left = pw_right = padding[1]

    images_padded = np.pad(
        images,
        ((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant',
        constant_values=0
    )

    output_h = int((h + ph_top + ph_bottom - kh) / sh) + 1
    output_w = int((w + pw_left + pw_right - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            image_slice = images_padded[:,
                                        i * sh: i * sh + kh,
                                        j * sw: j * sw + kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
