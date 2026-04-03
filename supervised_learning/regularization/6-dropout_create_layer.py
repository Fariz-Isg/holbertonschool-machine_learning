#!/usr/bin/env python3
"""Dropout Create Layer module"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer with Dropout using TensorFlow"""
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg")
    )
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )
    dropout = tf.keras.layers.Dropout(rate=1.0 - keep_prob)

    return dropout(dense(prev), training=training)
