#!/usr/bin/env python3
"""Module to change image contrast."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjusts the contrast of an image."""
    return tf.image.random_contrast(image, lower, upper)
