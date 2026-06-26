#!/usr/bin/env python3
"""
Define WGAN with gradient penalty and weight replacement
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with gradient penalty.

    Extends the basic Wasserstein GAN (Arjovsky et al. 2017) with the
    gradient penalty term proposed by Gulrajani et al. (2017). Instead
    of clipping weights, a penalty on the gradient norm of the discriminator
    is added to the discriminator loss. Also supports loading pre-trained
    weights from .h5 files.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initialize the WGAN_GP model.

        Args:
            generator (keras.Model): The generator neural network.
            discriminator (keras.Model): The discriminator (critic) network.
            latent_generator (Callable): Function that generates latent
                vectors given a batch size.
            real_examples (tf.Tensor): Dataset of real samples for training.
            batch_size (int, optional): Number of samples per training batch.
                Defaults to 200.
            disc_iter (int, optional): Number of discriminator update steps
                per generator update. Defaults to 2.
            learning_rate (float, optional): Learning rate for Adam optimizers.
                Defaults to 0.005.
            lambda_gp (float, optional): Coefficient for the gradient penalty
                term. Defaults to 10.
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda fake_output: -tf.reduce_mean(fake_output)
        self.generator.optimizer = (
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1,
                                  beta_2=self.beta_2))
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda real_output, fake_output: \
            tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        self.discriminator.optimizer = (
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1,
                                  beta_2=self.beta_2))
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

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate an interpolated sample between real and fake samples.

        Used for computing the gradient penalty. The interpolation
        coefficient u is sampled uniformly from [0, 1].

        Args:
            real_sample (tf.Tensor): A batch of real data samples.
            fake_sample (tf.Tensor): A batch of generated (fake) samples.

        Returns:
            tf.Tensor: Interpolated samples of the same shape as the inputs.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty for the discriminator.

        Enforces the Lipschitz constraint by penalizing the discriminator
        when its gradient norm deviates from 1 on interpolated samples.

        Args:
            interpolated_sample (tf.Tensor): Samples interpolated between
                real and fake data.

        Returns:
            tf.Tensor: Scalar gradient penalty value.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))

        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Custom training step for WGAN with gradient penalty.

        Performs disc_iter discriminator updates (each including a gradient
        penalty term) and then one generator update per call.

        Args:
            useless_argument: Required by Keras API but not used; real
                examples are accessed via self.real_examples.

        Returns:
            dict: A dictionary with keys 'discr_loss', 'gen_loss', and 'gp'
                containing the discriminator loss (without GP), generator
                loss, and gradient penalty for this step.
        """
        for _ in range(self.disc_iter):

            with (tf.GradientTape() as disc_tape):
                # get a real sample
                real_sample = self.get_real_sample()
                # get a fake sample
                fake_sample = self.get_fake_sample()
                # get the interpolated sample (between real and fake)
                interpoled_sample = self.get_interpolated_sample(real_sample,
                                                                 fake_sample)
                disc_real_output = self.discriminator(real_sample)
                disc_fake_output = self.discriminator(fake_sample)
                old_discr_loss = self.discriminator.loss(
                    real_output=disc_real_output,
                    fake_output=disc_fake_output)

                # compute the gradient penalty gp
                gp = self.gradient_penalty(interpoled_sample)

                # compute the sum new =discr_loss+self.lambda_gp*gp
                new_discr_loss = old_discr_loss + gp * self.lambda_gp

            disc_gradients = (
                disc_tape.gradient(new_discr_loss,
                                   self.discriminator.trainable_variables))
            self.discriminator.optimizer.apply_gradients(zip(
                disc_gradients,
                self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            # get a fake sample
            fake_sample = self.get_fake_sample(training=True)
            gen_out = self.discriminator(fake_sample, training=False)
            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(gen_out)

        gen_gradients = gen_tape.gradient(gen_loss,
                                          self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(
            gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": old_discr_loss, "gen_loss": gen_loss, "gp": gp}

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replace the weights of the generator and discriminator from .h5 files.

        Loads pre-trained weights from HDF5 files and assigns them to the
        generator and discriminator models, enabling use of models trained
        elsewhere (e.g., for longer epochs) without retraining.

        Args:
            gen_h5 (str): Path to the HDF5 (.h5) file containing the
                pre-trained generator weights.
            disc_h5 (str): Path to the HDF5 (.h5) file containing the
                pre-trained discriminator weights.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
