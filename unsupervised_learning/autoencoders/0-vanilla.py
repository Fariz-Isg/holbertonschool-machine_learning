#!/usr/bin/env python3
"""Vanilla Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a vanilla autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of nodes for each hidden layer in encoder
        latent_dims: integer, dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
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
