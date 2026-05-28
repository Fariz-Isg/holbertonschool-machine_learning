#!/usr/bin/env python3
"""
Contains the BayesianOptimization class that performs Bayesian Optimization
on a 1D black-box function.
"""
import numpy as np
from scipy.stats import norm


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian Process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the RBF kernel matrix between two matrices
        """
        a = np.sum(X1**2, 1).reshape(-1, 1)
        b = np.sum(X2**2, 1)
        sqdist = a + b - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a GP
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu_s.reshape(-1), np.diag(cov_s)

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process with new data points
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)


class BayesianOptimization:
    """
    Performs Bayesian Optimization on a 1D black-box function
    """

    def __init__(self, f, X_init, Y_init, bounds, acq_samples,
                 l=1, sigma_f=1, reference=0, xsi=0.01):
        """
        Class constructor
        """
        self.f = f
        self.gp = GaussianProcess(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], acq_samples
        ).reshape(-1, 1)
        self.xsi = xsi

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement
        """
        mu, sigma = self.gp.predict(self.X_s)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        Y_sample_opt = np.min(self.gp.Y)

        with np.errstate(divide='warn'):
            imp = Y_sample_opt - mu - self.xsi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei.reshape(-1)

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function using Expected Improvement
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        idx = np.argmin(self.gp.Y)
        return self.gp.X[idx], self.gp.Y[idx]
