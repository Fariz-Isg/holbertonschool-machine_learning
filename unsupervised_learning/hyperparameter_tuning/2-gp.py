#!/usr/bin/env python3
"""Gaussian Process module"""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initializes the Gaussian process."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the RBF covariance kernel matrix."""
        d = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(
            X2 ** 2, axis=1).reshape(1, -1) - 2 * X1 @ X2.T
        return self.sigma_f ** 2 * np.exp(-d / (2 * self.l ** 2))

    def predict(self, X_s):
        """Predicts the mean and variance of points in the Gaussian process."""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = (K_s.T @ K_inv @ self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T @ K_inv @ K_s)
        return mu, sigma

    def update(self, X_new, Y_new):
        """Updates the Gaussian process with a new sample point."""
        self.X = np.append(self.X, X_new.reshape(1, 1), axis=0)
        self.Y = np.append(self.Y, Y_new.reshape(1, 1), axis=0)
        self.K = self.kernel(self.X, self.X)
