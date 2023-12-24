from typing import Any
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

class Activation:
    
    def activate(self, x):
        pass
    
    def derivative(self, x):
        pass

class ReLU(Activation):
    
    def activate(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0) * 1

class Softmax(Activation):
    
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return x * (1 - x)
    
class Linear(Activation):
    
    def activate(self, x):
        return x
    
    def derivative(self, x):
        return 1
    
class Snake(Activation):
    
    def activate(self, x):
        return x + np.square(np.sin(x))
    
    def derivative(self, x):
        return 2 * np.cos(x) * np.sin(x) + 1
    




class Network:
    
    def __init__(self, dimensions, activations, alpha=0.1) -> None:
        self.layers = []
        self.activations = activations
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
            out = self.activations[layer].activate(net)#sigmoid(net)
            
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
        D = [ error * self.activations[-1].derivative(A[-1]) ]
        for layer in np.arange(len(A) - 2, 0, -1):
            
            # The delta for the current layer is equal to the delta of the
            # previous layer dotted with the weight matrix of the current layer
            # followed by multiplying the delta by the derivative of the nonlinear activation function
            delta = D[-1].dot(self.layers[layer].T)
            delta = delta * self.activations[layer].derivative(A[layer])
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
            p = self.activations[layer].activate(np.dot(p, self.layers[layer]))
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


POINTS = 100
SEQ_LEN = 20



def generate_split(x_raw, y_raw, seq_len: int, ratio: float) -> tuple:
    splice = int(seq_len * ratio)
    return x_raw[0:splice], y_raw[0:splice], x_raw[splice:seq_len], y_raw[splice:seq_len]


# Generate testing data
x_raw = np.linspace(0, SEQ_LEN, POINTS).reshape(-1, 1)
y_raw = (np.sin(x_raw*2))# This is our transform function
# print(np.max(y_raw))
# exit()

EPOCHS = 50000

# Generate splits
x_train, y_train, x_test, y_test = generate_split(x_raw, y_raw, POINTS, 0.6)
#print(x_train)
net = Network(
    dimensions=[1, 128, 1],
    activations=[Linear(), Snake(),Snake(), Linear()],
    alpha=0.000001
)
net.train(x_train, y_train, EPOCHS, batch_size=64, step=100)



# print(net.predict([1])
# )

plt.title("Regression Models")
plt.xlabel("X-Value")
plt.ylabel("Y-Value")
plt.scatter(x_raw, y_raw, color='green', label='Experimental Data')
plt.plot(x_raw, net.predict(x_raw), color='red', label='Simple Linear')
plt.legend(loc='upper left')
plt.show()
    
        