import numpy as np
import matplotlib.pyplot as plt

# Constants
SEQ_LEN: int = 50
POINTS: int = 50 + 1
WAV_AMPLITUDE: int = 3


RATIO = 0.8

def ssr(observed, predicted):
    return ((observed - predicted) ** 2).sum()

def dssr(observed, predicted):
    # The derivative of the SSR
    return (-2*(observed - predicted)).sum()

def predict(x, slope, intercept):
    return x * slope + intercept


def generate_split(x_raw, y_raw, seq_len: int, ratio: float) -> tuple:
    splice = int(seq_len * ratio)
    return x_raw[0:splice], y_raw[0:splice], x_raw[splice:seq_len], y_raw[splice:seq_len]


# Generate testing data
x_raw = np.linspace(0, SEQ_LEN, POINTS)
y_raw = np.sin(x_raw / 4) * WAV_AMPLITUDE + 0.2 * x_raw + 2

# Generate splits
x_train, y_train, x_test, y_test = generate_split(x_raw, y_raw, POINTS, RATIO)

LEARNING_RATE: float = 1e-3
EPOCHS = 10

slope = 0.2
intercept = 0
for step in range(EPOCHS):
    # Calculate the derivative
    dssr_val = dssr(y_train, predict(x_train, slope, intercept))
    
    # Calculate the step size
    step_size = dssr_val * LEARNING_RATE
    
    # Calculate the new slope (update)
    intercept = intercept - step_size
    print(f'({step}) dSSR: {dssr_val:.3f}\tStep: {step_size:+.3f}\tIntercept: {intercept:.2f}')

plt.title('One-Parameter Gradient Descent')
plt.plot(x_train, y_train, label='Real')
plt.plot(x_raw, predict(x_raw, slope, intercept), label='Trained')
plt.legend(loc='upper left')
plt.show()