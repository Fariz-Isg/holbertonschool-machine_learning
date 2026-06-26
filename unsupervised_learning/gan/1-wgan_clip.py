#!/usr/bin/env python3
"""Wasserstein GAN with weight clipping"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN with weight clipping.

    Implements the original Wasserstein GAN (Arjovsky et al. 2017) that
    enforces the Lipschitz constraint by clipping discriminator weights
    to a fixed range [-1, 1]. Compared to a Simple GAN, the losses are
    changed to the Wasserstein distance formulation.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initialize the WGAN_clip model.

        Args:
            generator (keras.Model): The generator neural network.
            discriminator (keras.Model): The discriminator (critic) network.
            latent_generator (Callable): Function that returns latent vectors
                given a batch size.
            real_examples (tf.Tensor): Dataset of real samples for training.
            batch_size (int, optional): Number of samples per training batch.
                Defaults to 200.
            disc_iter (int, optional): Number of discriminator update steps
                per generator update. Defaults to 2.
            learning_rate (float, optional): Learning rate for Adam optimizers.
                Defaults to 0.005.
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5   # standard value, but can be changed if necessary
        self.beta_2 = .9   # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        # Generator loss: opposite of mean discriminator output on fake samples
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        # Discriminator loss: mean on fake minus mean on real
        # (we want to maximize real - fake, so minimize fake - real)
        self.discriminator.loss = (
            lambda x, y: tf.math.reduce_mean(y) - tf.math.reduce_mean(x)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generate a batch of fake samples using the generator.

        Args:
            size (int, optional): Number of samples to generate.
                Defaults to self.batch_size.
            training (bool, optional): Whether the generator is in training
                mode. Defaults to False.

        Returns:
            tf.Tensor: A batch of generated (fake) samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Sample a batch of real examples from the dataset.

        Args:
            size (int, optional): Number of real samples to return.
                Defaults to self.batch_size.

        Returns:
            tf.Tensor: A batch of real samples randomly selected from
                the dataset.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Custom training step for WGAN with weight clipping.

        Performs disc_iter discriminator updates (each followed by weight
        clipping to [-1, 1]) and then one generator update per call.

        Args:
            useless_argument: Required by Keras API but not used; real
                examples are accessed via self.real_examples.

        Returns:
            dict: A dictionary with keys 'discr_loss' and 'gen_loss'
                containing the average discriminator loss and the generator
                loss for this step.
        """
        # Train discriminator disc_iter times
        for _ in range(self.disc_iter):
            # Compute the loss for the discriminator in a tape watching
            # the discriminator's weights
            with tf.GradientTape() as tape:
                # get a real sample
                real_sample = self.get_real_sample()
                # get a fake sample
                fake_sample = self.get_fake_sample(training=False)
                # compute the discriminator outputs
                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)
                # compute the loss: mean(fake) - mean(real)
                discr_loss = self.discriminator.loss(real_output, fake_output)

            # apply gradient descent once to the discriminator
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

            # clip the weights of the discriminator between -1 and 1
            for w in self.discriminator.trainable_variables:
                w.assign(tf.clip_by_value(w, -1.0, 1.0))

        # Compute the loss for the generator in a tape watching the
        # generator's weights
        with tf.GradientTape() as tape:
            # get a fake sample
            fake_sample = self.get_fake_sample(training=True)
            # pass through discriminator (not in training mode)
            fake_output = self.discriminator(fake_sample, training=False)
            # compute the loss: -mean(disc(fake))
            gen_loss = self.generator.loss(fake_output)

        # apply gradient descent to the generator
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
