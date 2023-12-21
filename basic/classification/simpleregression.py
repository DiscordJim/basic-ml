import numpy as np
import matplotlib.pyplot as plt

# Constants
SEQ_LEN: int = 50
POINTS: int = 1000
WAV_AMPLITUDE: int = 3

# Generate testing data
x: np.array[np.float32] = np.linspace(0, SEQ_LEN, POINTS)
y: np.array[np.float32] = np.sin(x) * WAV_AMPLITUDE + x


plt.plot(x, y)
plt.show()