#!/usr/bin/env python3
"""Module for calculating specificity from a confusion matrix."""
import numpy as np


def specificity(confusion):
    """Calculate specificity for each class in a confusion matrix."""
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = np.sum(confusion) - tp - fp - fn
    return tn / (tn + fp)
