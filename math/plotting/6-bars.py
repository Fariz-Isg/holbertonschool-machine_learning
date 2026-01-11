#!/usr/bin/env python3
"""
Stacked bar graph of fruit per person
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph of fruit per person
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    items = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    width = 0.5

    bottom = np.zeros(3)
    for i in range(4):
        plt.bar(people, fruit[i], width, bottom=bottom,
                color=colors[i], label=items[i])
        bottom += fruit[i]

    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.show()
