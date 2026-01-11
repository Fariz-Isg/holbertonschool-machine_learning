#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.8-prune
"""


def prune(df):
    """
    Docstring for prune

    :param df: Description
    :return: Description
    :rtype: Any
    """
    df = df.dropna(subset=['Close'])
    return df
