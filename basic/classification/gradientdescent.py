import numpy as np
import matplotlib.pyplot as plt

# Constants
SEQ_LEN: int = 50
POINTS: int = 50 + 1
WAV_AMPLITUDE: int = 3


RATIO = 0.8

def predict(x) -> float:
    return x

def calculate_r_squared(predicted, validation):  
    mean_value = np.mean(predicted)
    sum_of_residuals_mean = ((validation - mean_value)**2).sum()
    sum_of_residuals_fit = ((validation - predicted)**2).sum()
    return (sum_of_residuals_mean - sum_of_residuals_fit)/sum_of_residuals_mean
    

def generate_split(x_raw, y_raw, seq_len: int, ratio: float) -> tuple:
    splice = int(seq_len * ratio)
    return x_raw[0:splice], y_raw[0:splice], x_raw[splice:seq_len], y_raw[splice:seq_len]



# Generate testing data
x_raw = np.linspace(0, SEQ_LEN, POINTS)
y_raw = np.sin(x_raw / 4) * WAV_AMPLITUDE + 0.2 * x_raw

# Generate splits
x_train, y_train, x_test, y_test = generate_split(x_raw, y_raw, POINTS, RATIO)

fit_line = x_raw

prediction = predict(x_test)

print(calculate_r_squared(prediction, y_test))



plt.plot(x_train, y_train)
plt.plot(x_raw, fit_line)
plt.show()