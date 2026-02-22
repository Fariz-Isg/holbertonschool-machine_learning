#!/usr/bin/env python3
"""Module for creating a confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create and return a confusion matrix from one-hot labels and logits."""
    return np.dot(labels.T, logits)
