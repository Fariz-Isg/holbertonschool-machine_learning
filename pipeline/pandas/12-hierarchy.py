#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.12-hierarchy
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two DataFrames with MultiIndex hierarchy.

    : param df1: pd.DataFrame (coinbase)
    :param df2: pd.DataFrame (bitstamp)
    :return: concatenated pd.DataFrame with rearranged MultiIndex
    """

    # Index both dataframes on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Filter both dataframes:  timestamps from 1417411980 to 1417417980
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Concatenate with keys (df2 first = bitstamp, df1 = coinbase)
    result = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Rearrange MultiIndex:  Timestamp first, then bitstamp/coinbase
    result = result.swaplevel(0, 1)

    # Sort by Timestamp for chronological order
    result = result.sort_index()

    return result
