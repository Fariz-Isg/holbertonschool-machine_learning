#!/usr/bin/env python3
"""Module for calculating sensitivity from a confusion matrix."""
import numpy as np


def sensitivity(confusion):
    """Calculate sensitivity for each class in a confusion matrix."""
    return np.diag(confusion) / np.sum(confusion, axis=1)
