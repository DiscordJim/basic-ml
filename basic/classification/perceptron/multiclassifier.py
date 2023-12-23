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

Y = np.array([[0, 0, 1],
             [0, 1, 0],
             [0, 1, 0],
             [1, 0, 0]])

EPOCHS = 60
LEARNING_RATE = 1

# Intialize weights to random
# The sqrt helps converging faster
w = np.sqrt(np.random.uniform(size=(X.shape[1], Y.shape[1])))

def softmax(x):
    exp_x = np.exp(x - max(x))
    return exp_x / np.sum(exp_x)

# Amount of training examples
N = len(X)

# Run gradient descent
for e in range(EPOCHS): 
    total_loss = 0 
    for i in range(len(X)): # go through each training example
        # Specify our training case
        x, t = X[i], Y[i]

        # Calculate activation function over prediction
        y = softmax(w.T @ x)

        # Update the total loss with cross entropy loss
        # https://stats.stackexchange.com/questions/500902/clarification-needed-on-gradients-in-backpropagation
        gradient = np.outer(x, y - t)
        w = w - LEARNING_RATE * gradient / N


predict = lambda i : np.argmax(softmax(w.T @ np.array(i)))

print(f'Predictions:')
print(predict([1, 0, 0]))
print(predict([1, 1, 0]))
print(predict([1, 0, 1]))
print(predict([1, 1, 1]))

