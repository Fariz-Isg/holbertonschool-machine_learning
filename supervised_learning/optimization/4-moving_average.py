#!/usr/bin/env python3
"""Moving average module"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set"""
    v = 0
    moving_averages = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        bias_correction = 1 - (beta ** (i + 1))
        moving_averages.append(v / bias_correction)
    return moving_averages
