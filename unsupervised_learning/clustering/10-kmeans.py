#!/usr/bin/env python3
"""K-means clustering using sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset."""
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
