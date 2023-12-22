import numpy as np
import matplotlib.pyplot as plt
from random import uniform, randint

# Constants
SEQ_LEN: int = 10
POINTS: int = 20
WAV_AMPLITUDE: int = 4


# Hopefully this helps show the pattern :)
# https://towardsdatascience.com/polynomial-regression-gradient-descent-from-scratch-279db2936fe9

RATIO = 0.8

NOISE_AMPLITUDE = 3

def generate_split(x_raw, y_raw, seq_len: int, ratio: float) -> tuple:
    splice = int(seq_len * ratio)
    return x_raw[0:splice], y_raw[0:splice], x_raw[splice:seq_len], y_raw[splice:seq_len]


def predict(x, param_set: list):
    val = 0
    for i in range(len(param_set)):
        val += param_set[i] * (x ** i)
    return val

# Generate testing data
x_raw = np.linspace(0, SEQ_LEN, POINTS)
y_raw = -(x_raw - 3) ** 2 + 4
y_raw += np.random.normal(-NOISE_AMPLITUDE, NOISE_AMPLITUDE, POINTS)

# Generate splits
x_train, y_train, x_test, y_test = generate_split(x_raw, y_raw, POINTS, RATIO)

LEARNING_RATE: float = 0.0007
EPOCHS = 10000
MINI_BATCH_SIZE = 12

PARAMETERS = 3

weights = np.random.rand(PARAMETERS)

for step in range(EPOCHS):
    batch_x = []
    batch_y = []
    for i in range(MINI_BATCH_SIZE):
        sel = randint(0, len(x_train) - 1)
        batch_x.append(x_train[sel])
        batch_y.append(y_train[sel])
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    # Calculate total error with MSE
    e_total = ((1/2)*((batch_y - (predict(batch_x, weights)))**2)).sum()
    print(f'({step}) loss={e_total:.2f}\r',end='')
    
    # Do the gradient update step
    for d in range(len(weights)):
        # Calculate the gradient w/ partial derivates
        dw = (-2 / len(batch_x)) * ((batch_x ** d )*(batch_y - predict(batch_x, weights))).sum()
        
        # Update the graient
        weights[d] = weights[d] - (LEARNING_RATE * dw)
print()

plt.title('Multi-Parameter Stochastic Gradient Descent')
plt.scatter(x_raw, y_raw, label='Real')
plt.plot(x_raw, predict(x_raw, weights), color='red', label='Prediction')
plt.legend(loc='upper left')
plt.show()