#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.7-high
"""


def high(df):
    """
    Docstring for high

    :param df: Description
    """
    df = df.sort_values('High', ascending=False)
    return df
