#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of nodes for each hidden layer in encoder
        latent_dims: integer, dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick."""
        z_mean, z_log_var = args
        eps = keras.backend.random_normal(shape=keras.backend.shape(z_mean))
        return z_mean + keras.backend.exp(z_log_var / 2) * eps

    z = keras.layers.Lambda(sampling)([mu, log_var])
    encoder = keras.Model(inputs, [z, mu, log_var])

    dec_inputs = keras.Input(shape=(latent_dims,))
    y = dec_inputs
    for nodes in reversed(hidden_layers):
        y = keras.layers.Dense(nodes, activation='relu')(y)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(y)
    decoder = keras.Model(dec_inputs, output)

    auto_input = keras.Input(shape=(input_dims,))
    z_out, mu_out, log_var_out = encoder(auto_input)
    auto_output = decoder(z_out)
    auto = keras.Model(auto_input, auto_output)

    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var_out - keras.backend.square(mu_out)
        - keras.backend.exp(log_var_out), axis=1
    )
    auto.add_loss(keras.backend.mean(kl_loss))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
