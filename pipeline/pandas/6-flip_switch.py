#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.6-flip_switch
"""


def flip_switch(df):
    """
    Docstring for flip_switch

    :param df: Description
    """
    df = df.sort_index(ascending=False)
    df = df.transpose()
    return df
