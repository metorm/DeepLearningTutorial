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

    def BP(self,eta, RefResults, previous_layer_out):
        # return delta using log-likehood cost function
        # and update weights / biases
        # 'next_layer_delta' value is not used, just to keep the same API for different layers
        self.delta = [(a-y) for a,y in zip(self.out,RefResults)]
        nabla_w = [np.dot(a.reshape(len(a),1),d.reshape(1,len(d))) for d,a in zip(self.delta,previous_layer_out)]
        nabla_w = np.sum(nabla_w,axis=0) / len(nabla_w)
        nabla_b = np.sum(self.delta,axis=0) / len(self.delta)

        # update network parameters
        self.weights = self.weights - eta * nabla_w
        self.biases = self.biases - eta * nabla_b

        return self.delta