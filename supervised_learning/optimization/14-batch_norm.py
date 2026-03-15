#!/usr/bin/env python3
"""Batch Normalization Upgrade module"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    z = dense(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        gamma_initializer='ones',
        beta_initializer='zeros',
        epsilon=1e-7
    )
    z_norm = batch_norm(z)

    if activation is None:
        return z_norm
    return activation(z_norm)
