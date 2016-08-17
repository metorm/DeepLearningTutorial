import numpy as np

class Network(Object):

    def __init__(self,sizes):
        # sizes: array, numbers of neurons at each layer
        self.NumOfLayers = len(sizes)
        self.sizes = sizes
        # weights: between two neighbour layers there is a weight matrixe with size (n+1 th layer
        # size, n th layer size)
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        # input layer (0th) has no bias
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]

    def FeedForward(self,a):
        # iteratively calculate output of a neural network
        # No need to save intermedia values, because this function is only used for testing
        for w,b in zip(self.weights,self.biases):
            a = sigmoid(np.dot(w,a)+b) # matrix multiply
        return a

# sigmoid functions
# sigmoid
def sigmoid(z):
    # input can be signle number or array
    return 1.0/(1.0+np.exp(-z))
# devivate of sigmoid
def sigmoid_d(z):
    r = sigmoid(z)
    return r*(1-r)
