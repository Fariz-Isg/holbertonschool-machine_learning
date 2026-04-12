#!/usr/bin/env python3
"""Module to change image hue."""
import tensorflow as tf


def change_hue(image, delta):
    """Changes the hue of an image."""
    return tf.image.adjust_hue(image, delta)
