import numpy as np
import matplotlib.pyplot as plt

# Constants
SEQ_LEN: int = 50
POINTS: int = 50 + 1
WAV_AMPLITUDE: int = 3


# Hopefully this helps show the pattern :)

RATIO = 0.8


def generate_split(x_raw, y_raw, seq_len: int, ratio: float) -> tuple:
    splice = int(seq_len * ratio)
    return x_raw[0:splice], y_raw[0:splice], x_raw[splice:seq_len], y_raw[splice:seq_len]


# Generate testing data
x_raw = np.linspace(0, SEQ_LEN, POINTS)
y_raw = np.sqrt((np.sin(x_raw / 4) * WAV_AMPLITUDE) + 0.2 * x_raw) + 1

# Generate splits
x_train, y_train, x_test, y_test = generate_split(x_raw, y_raw, POINTS, RATIO)

LEARNING_RATE: float = 0.00003
EPOCHS = 700

slope = 0.5
intercept = 0
for step in range(EPOCHS):
    # Calculate the derivative
    x0 = (-2 * (y_train - (intercept + (slope * x_train)))).sum()
    x1 = (-2 * x_train * (y_train - (intercept + (slope * x_train)))).sum()


    # Calculate the step size
    s0 = x0 * LEARNING_RATE
    s1 = x1 * LEARNING_RATE
    
    # Update parameters
    intercept = intercept - s0
    slope = slope - s1

plt.title('Two-Parameter Gradient Descent')
plt.plot(x_raw, y_raw, label='Real')
plt.plot(x_raw, x_raw * slope + intercept, label='Prediction')
plt.legend(loc='upper left')
plt.show()