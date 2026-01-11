#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.2-from_file
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Docstring for from_file

    :param filename: Description
    :param delimiter: Description
    """
    df = pd.read_csv(filename, sep=delimiter)
    return pd.DataFrame(df)
