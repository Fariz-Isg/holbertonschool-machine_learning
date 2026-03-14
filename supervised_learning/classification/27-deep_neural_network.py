#!/usr/bin/env python3
"""DeepNeuralNetwork module"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """DeepNeuralNetwork class"""

    def __init__(self, nx, layers):
        """Constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx))
            else:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1]))

            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the deep neural network
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
        Returns:
            the output of the neural network and the cache
        """
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]

            z = np.matmul(W, A_prev) + b
            if i == self.__L:
                t = np.exp(z)
                A = t / np.sum(t, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-z))
            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y: numpy.ndarray, shape (classes, m), containing correct labels
            A: numpy.ndarray, shape (classes, m), containing activated output
        Returns:
            the cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        Args:
            X: numpy.ndarray, shape (nx, m), containing input data
            Y: numpy.ndarray, shape (classes, m), containing correct labels
        Returns:
            the neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A == np.max(A, axis=0), 1, 0)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Args:
            Y: numpy.ndarray, shape (classes, m), containing correct labels
            cache: dictionary containing intermediary values of the network
            alpha: learning rate
        Returns:
            None (Updates the private attribute __weights)
        """
        m = Y.shape[1]

        # Start with the last layer
        dz = cache["A{}".format(self.__L)] - Y

        # Iterate backwards through the layers to calculate gradients
        for i in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(i - 1)]

            dw = (1 / m) * np.matmul(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            # Next layer's dz propagation uses actual weights BEFORE updating
            if i > 1:
                W = self.__weights["W{}".format(i)]
                dz = np.matmul(W.T, dz) * (A_prev * (1 - A_prev))

            self.__weights["W{}".format(i)] = (
                self.__weights["W{}".format(i)] - alpha * dw)
            self.__weights["b{}".format(i)] = (
                self.__weights["b{}".format(i)] - alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        Args:
            X: numpy.ndarray, shape (nx, m), containing input data
            Y: numpy.ndarray, shape (classes, m), containing correct labels
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: whether or not to print info about the training
            graph: whether or not to graph info about the training
            step: the step size for verbose and graph
        Returns:
            evalutation of the training data
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)

            if verbose and (i % step == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
            if graph and (i % step == 0 or i == iterations):
                costs.append(cost)
                steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            plt.plot(steps, costs, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
