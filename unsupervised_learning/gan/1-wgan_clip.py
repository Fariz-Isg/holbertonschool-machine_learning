#!/usr/bin/env python3
"""Simple GAN"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """
    A simple Generative Adversarial Network (GAN) implementation.

    This class integrates a generator and a discriminator, along with
    methods for sampling fake and real data. It extends `keras.Model`
    to provide a flexible GAN training structure.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 latent_generator,
                 real_examples,
                 batch_size: int = 200,
                 disc_iter: int = 2,
                 learning_rate: float = 0.005):
        """
        Initialize the GAN with generator, discriminator, and optimizers.

        Args:
            generator (keras.Model): The generator model.
            discriminator (keras.Model): The discriminator model.
            latent_generator (Callable): Function to generate latent vectors.
            real_examples (tf.Tensor): Dataset of real samples for training.
            batch_size (int, optional): Number of samples per batch.
            disc_iter (int, optional): Number of discriminator updates
            per step.
            learning_rate (float, optional): Optimizer learning rate.
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate

        self.beta_1 = 0.5
        self.beta_2 = 0.9

        self.generator.loss = (
            lambda x: tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        self.discriminator.loss = (
            lambda x, y: tf.keras.losses.MeanSquaredError()
            (x, tf.ones(x.shape))
            + tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size: int = None, training: bool = False):
        """
        Generate a batch of fake samples using the generator.

        Args:
            size (int, optional): Number of fake samples.
            training (bool, optional): Whether the generator is
            in training mode.

        Returns:
            tf.Tensor: A batch of generated (fake) samples.
        """
        if size is None:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size: int = None):
        """
        Sample a batch of real examples from the dataset.

        Args:
            size (int, optional): Number of real samples.

        Returns:
            tf.Tensor: A batch of real samples randomly selected
            from the dataset.
        """
        if size is None:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, data):
        """
        Custom training step for GAN.

        Args:
            data (Any): Input data (unused here).

        Returns:
            dict: Dictionary containing discriminator and generator losses.

        Raises:
            NotImplementedError: If the training step is not implemented.
        """
        del data
        # Train discriminator
        discr_loss = 0.0
        for _ in range(self.disc_iter):
            real = self.get_real_sample(self.batch_size)
            fake = self.get_fake_sample(self.batch_size, training=False)
            with tf.GradientTape() as tape:
                pred_real = self.discriminator(real, training=True)
                pred_fake = self.discriminator(fake, training=True)
                d_loss = self.discriminator.loss(pred_real, pred_fake)
            grads = tape.gradient(
                d_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))
            discr_loss += d_loss
        discr_loss /= tf.cast(self.disc_iter, tf.float32)
        with tf.GradientTape() as tape:
            fake = self.get_fake_sample(self.batch_size, training=True)
            pred = self.discriminator(fake, training=False)
            gen_loss = self.generator.loss(pred)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
