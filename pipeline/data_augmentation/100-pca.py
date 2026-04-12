#!/usr/bin/env python3
"""Module to perform PCA color augmentation."""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.
    """
    img = tf.cast(image, tf.float32) / 255.0
    flat_img = tf.reshape(img, [-1, 3])

    mean = tf.reduce_mean(flat_img, axis=0)
    centered = flat_img - mean

    N = tf.cast(tf.shape(centered)[0] - 1, tf.float32)
    cov = tf.tensordot(tf.transpose(centered), centered, 1) / tf.maximum(N, 1.)

    eig_val, eig_vec = tf.linalg.eigh(cov)

    delta = tf.tensordot(eig_vec, alphas * eig_val, 1)

    pca_img = img + delta
    pca_img = tf.clip_by_value(pca_img, 0., 1.)

    return tf.cast(pca_img * 255.0, tf.uint8)
