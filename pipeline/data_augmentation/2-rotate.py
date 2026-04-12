#!/usr/bin/env python3
"""Module to rotate an image."""
import tensorflow as tf


def rotate_image(image):
    """Rotates an image by 90 degrees counter-clockwise."""
    return tf.image.rot90(image)
