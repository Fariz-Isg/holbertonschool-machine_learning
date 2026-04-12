#!/usr/bin/env python3
"""Module to randomly crop an image."""
import tensorflow as tf


def crop_image(image, size):
    """Performs a random crop of an image."""
    return tf.image.random_crop(value=image, size=size)
