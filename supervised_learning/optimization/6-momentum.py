#!/usr/bin/env python3
"""Momentum Upgrade module"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm
    in TensorFlow
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
