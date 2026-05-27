# Clustering

This project implements clustering algorithms, beginning with K-means initialization.

## Tasks

### [0. Initialize K-means](0-initialize.py)
A function `def initialize(X, k):` that initializes cluster centroids for K-means:
- `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset.
- `k` is a positive integer containing the number of clusters.
- Centroids are initialized with a multivariate uniform distribution along each dimension.
- Returns a `numpy.ndarray` of shape `(k, d)` containing the initialized centroids, or `None` on failure.