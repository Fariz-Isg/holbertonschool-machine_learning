#!/usr/bin/env python3
"""
Start by creating a function def from_numpy(array):
that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray
    """
    n = array.shape[1]
    columns = [chr(65 + i) for i in range(n)]
    return pd.DataFrame(array, columns=columns)
