#!/usr/bin/env python3
"""
Binomial distribution class
"""


class Binomial:
    """
    Represents a Binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            self.n = int(round(mean / p))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes"
        """
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        factorial_n = 1
        for i in range(1, self.n + 1):
            factorial_n *= i

        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        factorial_nk = 1
        for i in range(1, self.n - k + 1):
            factorial_nk *= i

        combination = factorial_n / (factorial_k * factorial_nk)

        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))
