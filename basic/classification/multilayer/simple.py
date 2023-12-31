import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt
from sklearn.utils import shuffle

# useful articles
# implementation heavily follows from the below
# -https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
# https://pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/
# Helper functions
def relu(x):
    return np.maximum(0, x)
    #return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    return (x > 0) * 1

def sigmoid(x):
    #return np.maximum(0, x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    #return (x > 0) * 1
    return x * (1 - x)


class Network:
    
    def __init__(self, dimensions, alpha=0.1) -> None:
        self.layers = []
        self.alpha = alpha
        for i in np.arange(0, len(dimensions) - 2):
            # We add an extra one to account for bias
            w = np.random.randn(dimensions[i] + 1, dimensions[i + 1] + 1)
            self.layers.append(w / np.sqrt(dimensions[i]))
        # The last two layers do not need a bias term
        w= np.random.randn(dimensions[-2] + 1, dimensions[-1])
        self.layers.append(w / np.sqrt(dimensions[-2]))
        
    def update_gradient(self, x, y):
        # Firstly, we ensure that our matrix is a numpy matrix
        # and has reasonable dimensions.
        A = [np.atleast_2d(x)]

        # This is the forward pass, hence why we are iterating forward
        # through the layers (duh!)
        for layer in np.arange(0, len(self.layers)):
            
            # Dot product between the activation and the weight matrix
            # this is the 'net input'
            net = A[layer].dot(self.layers[layer])
            
            # Apply the sigmoid activation function to calculate 
            # the output.
            out = sigmoid(net)
            
            # Record the activation
            A.append(out)
        # More aptly, the above could be described as forward propagation.
        
        # BACKPROP
        
        # Final output and the true target value
        # If this is not clear, think about how 
        # the forward propagation step functions.
        error = A[-1] - y 
        
        # Backward pass
        # We do ths by multiplying the derivative acro
        # Please note! We are moving through the layers backwards,
        # how interesting.
        #print(sigmoid_derivative(A[-1]))
        D = [ error * sigmoid_derivative(A[-1]) ]
        for layer in np.arange(len(A) - 2, 0, -1):
            
            # The delta for the current layer is equal to the delta of the
            # previous layer dotted with the weight matrix of the current layer
            # followed by multiplying the delta by the derivative of the nonlinear activation function
            delta = D[-1].dot(self.layers[layer].T)
            delta = delta * sigmoid_derivative(A[layer])
            D.append(delta)
            
        # Update weights
        D = D[::-1]
        
        for layer in np.arange(0, len(self.layers)):
            self.layers[layer] += 0.99 * -self.alpha * A[layer].T.dot(D[layer])
            
    
    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)
        
        if add_bias:
            p = np.c_[p, np.ones((p.shape[0]))]
        
        for layer in np.arange(0, len(self.layers)):
            p = sigmoid(np.dot(p, self.layers[layer]))
        return p
    
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss
    
    def next_batch(self, x, y, batch_size):
        for i in np.arange(0, x.shape[0], batch_size):
            yield (x[i:i + batch_size], y[i:i + batch_size])
    
    def train(self, X, Y, epochs, batch_size=16, step=100):
        # Adding bias
        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in range(epochs):
            X, Y = shuffle(X, Y)
            # Stochastic gradient descent
            for (x, target) in self.next_batch(X, Y, batch_size):
                self.update_gradient(x, target)
            
            if epoch % step == 0:
                print(f'Epoch: {epoch} -> {self.calculate_loss(X, Y)}')
            
    def __repr__(self) -> str:
        return 'NN'

# print(sigmoid_derivative(np.array([-0.3, 0.3, -0.1, 1])))

# print(np.max(np.array([[1],[2],[3]])))
# exit()

POINTS = 80
SEQ_LEN = 10



def generate_split(x_raw, y_raw, seq_len: int, ratio: float) -> tuple:
    splice = int(seq_len * ratio)
    return x_raw[0:splice], y_raw[0:splice], x_raw[splice:seq_len], y_raw[splice:seq_len]


# Generate testing data
x_raw = np.linspace(0, SEQ_LEN, POINTS).reshape(-1, 1)
y_raw = (x_raw ** np.sin(x_raw)) / 7.86#(0.5 * (np.sin(x_raw * 10) + 1) + np.sqrt(x_raw))/2.25# This is our transform function
# print(np.max(y_raw))
# exit()

EPOCHS = 10000

# Generate splits
x_train, y_train, x_test, y_test = generate_split(x_raw, y_raw, POINTS, 0.9)
#print(x_train)
net = Network([1, 5, 1], alpha=0.1)
net.train(x_train, y_train, EPOCHS, step=1)



# print(net.predict([1])
# )

plt.title("Regression Models")
plt.xlabel("X-Value")
plt.ylabel("Y-Value")
plt.scatter(x_raw, y_raw, color='green', label='Experimental Data')
plt.plot(x_raw, net.predict(x_raw), color='red', label='Simple Linear')
plt.legend(loc='upper left')
plt.show()
    
        