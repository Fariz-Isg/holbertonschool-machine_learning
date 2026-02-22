#!/usr/bin/env python3
"""Module for calculating precision from a confusion matrix."""
import numpy as np


def precision(confusion):
    """Calculate precision for each class in a confusion matrix."""
    return np.diag(confusion) / np.sum(confusion, axis=0)
