import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import math


POINTS: int = 100
WAV_AMPLITUDE: int = 3


def binomial(x: int, n: int, p: float) -> float:
    """The Binomial Distribution

    Args:
        x (_type_): _description_
        n (_type_): The total 
        p (_type_): Probability of the outcome

    Returns:
        _type_: _description_
    """
    return (factorial(n)/(factorial(x)*factorial(n-x)))*((p**x) * ((1 - p)**(n-x)))


def poisson(x: int, l: float) -> float:
    """Poisson Distribution

    Args:
        x (float): how many things
        l (float): occurences / hr

    Returns:
        float: _description_
    """
    return ((math.e ** -l)*(l**x))/(factorial(x))

def normal(x, mean, stdev):
    """Normal Distribution

    Args:
        x (_type_): _description_
        mu (_type_): _description_
        si (_type_): _description_
    """
    return (np.pi * stdev) * np.exp(-0.5 * ((x - mean) / stdev)**2)

# Generate testing data
x = np.linspace(0, POINTS, POINTS + 1, dtype=np.int8)
y = normal(x, POINTS // 2, 5)


plt.plot(x, y)
plt.show()