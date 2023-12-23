import numpy as np

# useful articles
# implementation heavily follows from the below
# -https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
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
            self.layers[layer] += -self.alpha * A[layer].T.dot(D[layer])
            
    
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
    
    def train(self, X, Y, epochs, step=100):
        # Adding bias
        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in range(epochs):
            
            for (x, target) in zip(X, Y):
                self.update_gradient(x, target)
            
            if epoch % step == 0:
                print(f'Epoch: {epoch} -> {self.calculate_loss(X, Y)}')
            
    def __repr__(self) -> str:
        return 'NN'


    
    
net = Network([2, 10, 1])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

net.train(X, Y, 1)

print(net.predict([1, 1])
)
    
        