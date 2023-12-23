import numpy as np

# A great article on this.
# https://pyimagesearch.com/2021/05/06/implementing-the-perceptron-neural-network-with-python/

# Setup our label & target arrays,
# in this case our example is AND.
# The first column is for bias (x0)
X = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

Y = np.array([[0],
              [0],
              [0],
              [1]])

EPOCHS = 10
LEARNING_RATE = 0.1

# Intialize weights to random
# The sqrt helps converging faster
w = np.sqrt(np.random.uniform(size=X.shape[1]))


def step(x: float) -> int:
    """A step function

    Args:
        x (float): a floating point value

    Returns:
        int: the step value
    """
    return 1 if x > 0 else 0

def predict(x, w) -> int:
    """Makes a prediction based on the input data and the weights matrix

    Args:
        x (np.array): the input data
        w (np.array): the weights matrix

    Returns:
        int: the prediction of the perceptron node
    """
    return step(np.dot(x, w))

for e in range(EPOCHS): # Train for EPOCHS rounds
    for i in range(len(X)): # go through each training example
        # Specify our training case
        x, t = X[i], Y[i]
        
        p = predict(x, w) # Make a prediction
        if p != t:
            error = p - t # see the difference between the error
            w += -LEARNING_RATE * error * x # update the weights
            
# Test an example
print(step([1, 0, 1] @ w.T))
