#!/usr/bin/env python3
"""GMM clustering using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset."""
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    return (model.weights_, model.means_, model.covariances_,
            model.predict(X), model.bic(X))
