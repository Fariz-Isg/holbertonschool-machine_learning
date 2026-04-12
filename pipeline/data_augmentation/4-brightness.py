#!/usr/bin/env python3
"""Module to change image brightness."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image."""
    return tf.image.random_brightness(image, max_delta)
