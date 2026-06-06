#!/usr/bin/env python3
"""Sparse Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of nodes for each hidden layer in encoder
        latent_dims: integer, dimensions of the latent space
        lambtha: L1 regularization parameter on the encoded output

    Returns:
        encoder, decoder, auto
    """
    reg = keras.regularizers.L1(lambtha)

    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=reg
    )(encoded)
    encoder = keras.Model(inputs, latent)

    dec_inputs = keras.Input(shape=(latent_dims,))
    decoded = dec_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(dec_inputs, output)

    auto_input = keras.Input(shape=(input_dims,))
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
