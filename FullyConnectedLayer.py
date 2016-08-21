import numpy as np
from sigmoid import sigmoid
from sigmoid import sigmoid_d


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, activation_fn_d = sigmoid_d):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.activation_fn_d = activation_fn_d
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
        self.zs = [(np.dot(im,self.weights) + self.biases) for im in input_images]
        self.out = [self.activation_fn(z) for z in self.zs]
        return self.out

    def BP(self,eta, BP_delta, next_layer, previous_layer_out):
        # return delta of this layer
        # and update weights / biases
        delta = [np.dot(w.transpose(),d)*self.activation_fn_d(z) for w,d,z in zip(next_layer.weights,BP_delta,self.zs)]
        nabla_w = [np.dot(a.reshape(len(a),1),d.reshape(1,len(d))) for d,a in zip(delta,previous_layer_out)]
        nabla_w = np.sum(nabla_w,axis=0) / len(nabla_w)
        nabla_b = np.sum(delta,axis=0) / len(delta)

        # update network parameters
        self.weights = self.weights - eta * nabla_w
        self.biases = self.biases - eta * nabla_b

        return delta