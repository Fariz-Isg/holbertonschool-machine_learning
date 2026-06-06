#!/usr/bin/env python3
"""Convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    Args:
        input_dims: tuple of integers, dimensions of the model input
        filters: list of filters for each conv layer in the encoder
        latent_dims: tuple of integers, dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """
    inputs = keras.Input(shape=input_dims)
    encoded = inputs
    for f in filters:
        encoded = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding='same'
        )(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoder = keras.Model(inputs, encoded)

    dec_inputs = keras.Input(shape=latent_dims)
    decoded = dec_inputs
    rev_filters = list(reversed(filters))
    for i, f in enumerate(rev_filters):
        if i < len(rev_filters) - 1:
            decoded = keras.layers.Conv2D(
                f, (3, 3), activation='relu', padding='same'
            )(decoded)
        else:
            decoded = keras.layers.Conv2D(
                f, (3, 3), activation='relu', padding='valid'
            )(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    channels = input_dims[-1]
    output = keras.layers.Conv2D(
        channels, (3, 3), activation='sigmoid', padding='same'
    )(decoded)
    decoder = keras.Model(dec_inputs, output)

    auto_input = keras.Input(shape=input_dims)
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
