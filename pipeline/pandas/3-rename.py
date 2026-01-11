#!/usr/bin/env python3
"""
Docstring for pipeline.pandas.2-from_file
"""
import pandas as pd


def rename(df):
    """
    Docstring for rename

    :param df: Description
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
