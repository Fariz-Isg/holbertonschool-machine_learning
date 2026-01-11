#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.13-analyze
"""


def analyze(df):
    """
    Docstring for analyze

    :param df: Description
    """
    df = df.drop('Timestamp', axis=1)
    df = df.describe()
    return df
