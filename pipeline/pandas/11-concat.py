#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.11-concat
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two DataFrames with specific conditions.

    :param df1: pd. DataFrame (coinbase)
    :param df2: pd.DataFrame (bitstamp)
    :return: concatenated pd.DataFrame
    """

    # Index both dataframes on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Filter df2: timestamps up to and including 1417411920
    df2 = df2.loc[: 1417411920]

    # Concatenate with keys
    result = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    return result
