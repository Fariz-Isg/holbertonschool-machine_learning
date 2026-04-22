#!/usr/bin/env python3
"""ResNet-50 architecture."""
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture."""
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    # Stage 1
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                            kernel_initializer=init)(X)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(act1)

    # Stage 2
    X2 = projection_block(pool1, [64, 64, 256], s=1)
    X2 = identity_block(X2, [64, 64, 256])
    X2 = identity_block(X2, [64, 64, 256])

    # Stage 3
    X3 = projection_block(X2, [128, 128, 512], s=2)
    X3 = identity_block(X3, [128, 128, 512])
    X3 = identity_block(X3, [128, 128, 512])
    X3 = identity_block(X3, [128, 128, 512])

    # Stage 4
    X4 = projection_block(X3, [256, 256, 1024], s=2)
    X4 = identity_block(X4, [256, 256, 1024])
    X4 = identity_block(X4, [256, 256, 1024])
    X4 = identity_block(X4, [256, 256, 1024])
    X4 = identity_block(X4, [256, 256, 1024])
    X4 = identity_block(X4, [256, 256, 1024])

    # Stage 5
    X5 = projection_block(X4, [512, 512, 2048], s=2)
    X5 = identity_block(X5, [512, 512, 2048])
    X5 = identity_block(X5, [512, 512, 2048])

    # Average Pooling and Dense
    avg_pool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(X5)
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=output)
    return model
