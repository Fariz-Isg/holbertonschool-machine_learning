#!/usr/bin/env python3
"""Predict module"""


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network"""
    return network.predict(x=data, verbose=verbose)
