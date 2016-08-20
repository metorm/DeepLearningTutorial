import numpy as np
import sigmoid


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid.sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out))
        self.biases = np.random.normal(loc=0.0, scale=1.0, size=(n_out,))

    def calculate(self, input_images):
        """
        :param input_images: a batch of results obtained from a batch of input images
                            arranged by function 'flatten_poolingLayer_output'
                            1st dimension: each image in a batch
                            2ed dimension: flattened output contains all feature maps of an original image
        :return: output activation value of this layer
        """
        self.input_images = input_images
        return [self.activation_fn(np.dot(im,self.weights) + self.biases) for im in input_images]