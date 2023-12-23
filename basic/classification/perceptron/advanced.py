import numpy as np
import math 

# A great article on this.
# https://medium.com/@jaimin-k/minimizing-cross-entropy-loss-in-binary-classification-4166ae04a22a

# Setup our label & target arrays,
# in this case our example is AND.
# The first column is for bias (x0)
X = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

Y = np.array([0, 0, 0, 1])

EPOCHS = 200
LEARNING_RATE = 0.1

# Intialize weights to random
# The sqrt helps converging faster
w = np.sqrt(np.random.uniform(size=X.shape[1]))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x, w):
    return 1 if sigmoid(x @ w) > 0.5 else 0

# Amount of training examples
N = len(X)

# Run gradient descent
for e in range(EPOCHS): 
    print(f'Epoch: {e}')
    total_loss = 0 
    for i in range(len(X)): # go through each training example
        # Specify our training case
        x, t = X[i], Y[i]
        
        # Calculate activation function over prediction
        y = sigmoid(x @ w)
        
        # Update the total loss with cross entropy loss
        total_loss = total_loss + ((y - t) * x)
    
    # Update 
    w = w - (LEARNING_RATE * (total_loss / N))
 
print()
print(sigmoid(w.T @ np.array([1, 0, 1])))
print(sigmoid(w.T @ np.array([1, 1, 1])))
step = lambda x : 1 if x > 0.0 else 0
print(f'\nTesting on examples: ')
print(step(w.T @ np.array([1, 0, 0])))
print(step(w.T @ np.array([1, 1, 0])))
print(step(w.T @ np.array([1, 0, 1])))
print(step(w.T @ np.array([1, 1, 1])))

