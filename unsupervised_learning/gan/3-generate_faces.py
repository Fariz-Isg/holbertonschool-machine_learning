#!/usr/bin/env python3
"""Convolutional Generator and Discriminator for face generation"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def convolutional_GenDiscr():
    """
    Build and return a convolutional generator and discriminator for
    generating 16x16 greyscale face images.

    The generator takes a 16-dimensional latent vector and upsamples it
    through Conv2D layers to produce a (16, 16, 1) image.

    The discriminator takes a (16, 16, 1) image, applies successive Conv2D
    and MaxPooling layers, and outputs a single scalar value.

    Both models use 'tanh' activations and Conv2D layers with padding='same'.

    Returns:
        tuple: A tuple (generator, discriminator) where:
            - generator (keras.Model): Takes input of shape (16,) and outputs
              shape (16, 16, 1).
            - discriminator (keras.Model): Takes input of shape (16, 16, 1)
              and outputs shape (1,).
    """

    def get_generator():
        """
        Build the convolutional generator model.

        Architecture:
            Input (16,) -> Dense(2048) -> Reshape(2, 2, 512)
            -> UpSampling2D -> Conv2D(64) -> BatchNorm -> tanh
            -> UpSampling2D -> Conv2D(16) -> BatchNorm -> tanh
            -> UpSampling2D -> Conv2D(1)  -> BatchNorm -> tanh
            Output: (16, 16, 1)

        Returns:
            keras.Model: The generator model named 'generator'.
        """
        inputs = keras.Input(shape=(16,))

        # Dense to get enough features, then reshape to spatial
        x = keras.layers.Dense(2 * 2 * 512)(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # Upsample 2x2 -> 4x4
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Upsample 4x4 -> 8x8
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(16, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Upsample 8x8 -> 16x16
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(1, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)

        return keras.Model(inputs, outputs, name='generator')

    def get_discriminator():
        """
        Build the convolutional discriminator model.

        Architecture:
            Input (16, 16, 1)
            -> Conv2D(32) -> MaxPooling2D -> tanh
            -> Conv2D(64) -> MaxPooling2D -> tanh
            -> Conv2D(128) -> MaxPooling2D -> tanh
            -> Conv2D(256) -> MaxPooling2D -> tanh
            -> Flatten -> Dense(1, tanh)
            Output: (1,)

        Returns:
            keras.Model: The discriminator model named 'discriminator'.
        """
        inputs = keras.Input(shape=(16, 16, 1))

        # Conv block 1: 16x16 -> 8x8
        x = keras.layers.Conv2D(32, kernel_size=3, padding='same')(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Conv block 2: 8x8 -> 4x4
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Conv block 3: 4x4 -> 2x2
        x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Conv block 4: 2x2 -> 1x1
        x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Flatten and output
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        return keras.Model(inputs, outputs, name='discriminator')

    return get_generator(), get_discriminator()
