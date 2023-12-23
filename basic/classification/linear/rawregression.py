import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

# Good article on linear regression
# https://realpython.com/linear-regression-in-python/
# https://data36.com/polynomial-regression-python-scikit-learn/

# Constants
SEQ_LEN: int = 100
POINTS: int = 20
WAV_AMPLITUDE: int = 2

RATIO = 0.8


def generate_split(x_raw, y_raw, seq_len: int, ratio: float) -> tuple:
    splice = int(seq_len * ratio)
    return x_raw[0:splice], y_raw[0:splice], x_raw[splice:seq_len], y_raw[splice:seq_len]

def create_poly_fit(x, y, degree):
    # https://towardsdatascience.com/implementing-linear-and-polynomial-regression-from-scratch-f1e3d422e6b4
    # Here is an example with a degree one polynomial:
    # X = np.c_[np.square(x_train), x_train, np.ones(N)]

    # We do it backwards so we can wrap it with the poly1d
    X = np.stack([np.power(x, n) for n in range(degree, -1, -1)], axis=1)
    
    # Calculate the weights (W)
    # This formula is w = inv(X.T @ X) @ X.T @ y
    A = np.linalg.inv(X.T @ X)
    D = A @ X.T
    W = D @ y_train
    
    return np.poly1d(W)
    


# Generate testing data
x_raw = np.linspace(1, SEQ_LEN, POINTS)
y_raw = np.log(x_raw) + np.sin(x_raw) # This is our transform function

# Generate splits
x_train, y_train, x_test, y_test = generate_split(x_raw, y_raw, POINTS, RATIO)


# Create a degree polynomial and fit it
poly_2 = create_poly_fit(x_train, y_train, 1)
predictions = poly_2(x_raw)

# Display the plot
plt.title("Regression Models")
plt.xlabel("X-Value")
plt.ylabel("Y-Value")
plt.scatter(x_raw, y_raw, color='green', label='Experimental Data')
plt.plot(x_raw, predictions, color='red', label='Simple Linear')
plt.legend(loc='upper left')
plt.show()