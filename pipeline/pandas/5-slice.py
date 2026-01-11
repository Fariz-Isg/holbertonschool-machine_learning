#!/usr/bin/env python3
"""
Docstring for pipeline.5-slice
"""


def slice(df):
    """
    Docstring for slice

    :param df: Description
    """
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    df = df.iloc[::60]
    return df
