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

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        pi = 3.1415926536
        e = 2.7182818285
        z = (x - self.mean) / self.stddev

        coeff = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -0.5 * (z ** 2)

        return coeff * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        pi = 3.1415926536
        mean = self.mean
        stddev = self.stddev
        z = (x - mean) / stddev
        val = z / (2 ** 0.5)

        erf = (val - (val ** 3) / 3 + (val ** 5) / 10 -
               (val ** 7) / 42 + (val ** 9) / 216)
        erf *= 2 / (pi ** 0.5)

        return 0.5 * (1 + erf)
