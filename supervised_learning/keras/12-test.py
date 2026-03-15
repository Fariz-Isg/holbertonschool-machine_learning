#!/usr/bin/env python3
"""Test module"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network"""
    return network.evaluate(x=data, y=labels, verbose=verbose)
