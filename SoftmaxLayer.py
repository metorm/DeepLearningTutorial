import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        # Initialize weights and biases
        self.weights = np.zeros((n_in, n_out))
        self.biases = np.zeros(n_out)

    def calculate(self, input_images):
        self.input_images = input_images
        out = [(np.dot(im,self.weights) + self.biases) for im in input_images]
        self.out = [softmax(im) for im in out]
        return self.out

    def BP_initial_delta(self,RefResults):
        return [(a-y) for a,y in zip(self.out,RefResults)]