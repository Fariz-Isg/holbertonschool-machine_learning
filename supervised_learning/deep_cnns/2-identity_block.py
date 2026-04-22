#!/usr/bin/env python3
"""Identity block for ResNet."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds identity block as described in Deep Residual Learning."""
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # 1st component
    conv1 = K.layers.Conv2D(F11, (1, 1), padding='same',
                            kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # 2nd component
    conv2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            kernel_initializer=init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # 3rd component
    conv3 = K.layers.Conv2D(F12, (1, 1), padding='same',
                            kernel_initializer=init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add shortcut and activate
    add = K.layers.Add()([bn3, A_prev])
    output = K.layers.Activation('relu')(add)

    return output
