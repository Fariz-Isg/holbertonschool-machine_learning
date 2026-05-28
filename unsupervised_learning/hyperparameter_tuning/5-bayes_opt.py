#!/usr/bin/env python3
"""Bayesian Optimization module"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initializes Bayesian Optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location using Expected Improvement."""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            f_best = np.min(self.gp.Y)
            imp = f_best - mu - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            imp = mu - f_best - self.xsi
        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0] = 0
        X_next = self.X_s[np.argmax(EI)].reshape(-1)
        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function."""
        X_prev = None
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_prev is not None and np.isclose(X_next, X_prev):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            X_prev = X_next
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        return self.gp.X[idx].reshape(-1), self.gp.Y[idx].reshape(-1)
