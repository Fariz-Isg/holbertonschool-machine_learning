#!/usr/bin/env python3
"""Normalize module"""


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix"""
    return (X - m) / s
