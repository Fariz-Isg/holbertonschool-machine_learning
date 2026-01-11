#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.10-index
"""


def index(df):
    """
    Docstring for index

    :param df: Description
    """
    df = df.set_index('Timestamp')
    return df
