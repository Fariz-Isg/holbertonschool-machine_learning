#!/usr/bin/env python3
"""
Normal distribution class
"""


class Normal:
    """
    Represents a Normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            sum_squared_diff = sum((x - self.mean) ** 2 for x in data)
            self.stddev = float((sum_squared_diff / len(data)) ** 0.5)
