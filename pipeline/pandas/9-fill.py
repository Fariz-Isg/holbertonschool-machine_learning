#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.9-fill
"""


def fill(df):
    """
    Docstring for fill

    :param df: Description
    """
    df = df.drop('Weighted_Price', axis=1)
    df['Close'] = df['Close'].ffill()
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
