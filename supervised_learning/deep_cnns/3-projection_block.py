#!/usr/bin/env python3
"""Projection block for ResNet."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds projection block as described in Deep Residual Learning."""
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main path
    conv1 = K.layers.Conv2D(F11, (1, 1), strides=(s, s), padding='same',
                            kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    conv2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            kernel_initializer=init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    conv3 = K.layers.Conv2D(F12, (1, 1), padding='same',
                            kernel_initializer=init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Shortcut path
    conv_sh = K.layers.Conv2D(F12, (1, 1), strides=(s, s), padding='same',
                              kernel_initializer=init)(A_prev)
    bn_sh = K.layers.BatchNormalization(axis=3)(conv_sh)

    # Add and activate
    add = K.layers.Add()([bn3, bn_sh])
    output = K.layers.Activation('relu')(add)

    return output
