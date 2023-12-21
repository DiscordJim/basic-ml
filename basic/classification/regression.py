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


# Generate testing data
x_raw = np.linspace(1, SEQ_LEN, POINTS)
y_raw = np.log(x_raw) + np.sin(x_raw) # This is our transform function

# Generate splits
x_train, y_train, x_test, y_test = generate_split(x_raw, y_raw, POINTS, RATIO)

# Create models
model_basic = np.poly1d(np.polyfit(x_train, y_train, 1))
poly_low = np.poly1d(np.polyfit(x_train, y_train, 2))

# Create Predictions
prediction_basic = model_basic(x_raw)
prediction_poly_low = poly_low(x_raw)


# Display the plot
plt.title("Regression Models")
plt.xlabel("X-Value")
plt.ylabel("Y-Value")
plt.scatter(x_raw, y_raw, color='green', label='Experimental Data')
plt.plot(x_raw, prediction_basic, color='red', label='Simple Linear')
plt.plot(x_raw, prediction_poly_low, color='blue', label='Polynomial (2)')
plt.legend(loc='upper left')
plt.show()