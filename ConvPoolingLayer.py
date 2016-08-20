# a convolution layer with a following pooling layer

import numpy as np
import sigmoid
import conv2d


class ConvPoolingLayer(object):
    def __init__(self, input_shape, filter_shape, pool_size=(2, 2), activation_fn=sigmoid.sigmoid, use_max_pooling=True):
        # store input values

        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `input_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `pool_size` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """

        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.use_max_pooling = use_max_pooling
        self.activation_fn = activation_fn

        # number of outputs after pooling
        num_output = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size)
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / num_output), size=filter_shape)
        self.biases = np.random.normal(loc=0.0, scale=1.0, size=(filter_shape[0],))

    # accept input, and return result of forward propagation
    def calculate(self, input_images):
        self.input_images = input_images
        conv_out = conv2d.conv2d(self.input_images, self.weights, self.input_shape)
        # add bias to conv_out
        # for each image in a mini batch
        for mb in xrange(self.input_shape[0]):
            # for each kernel
            for k in xrange(self.filter_shape[0]):
                conv_out[mb][k] = np.asarray(conv_out[mb][k]) + self.biases[k]
        # pooling and return
        if self.use_max_pooling:
            return [[np.asarray(self.max_pooling(conv_out[mb][k])) for k in xrange(self.filter_shape[0])]
                    for mb in xrange(self.input_shape[0])]
        else:
            return [[np.asarray(self.mean_pooling(conv_out[mb][k])) for k in xrange(self.filter_shape[0])]
                    for mb in xrange(self.input_shape[0])]

    def max_pooling(self, in_array):
        in_array_h, in_array_w = in_array.shape
        return [[self.activation_fn(np.amax(in_array[i:i + self.pool_size[0], j:j + self.pool_size[1]]))
                 for i in xrange(0, in_array_h, self.pool_size[0])]
                for j in xrange(0, in_array_w, self.pool_size[1])]

    def mean_pooling(self, in_array):
        in_array_h, in_array_w = in_array.shape
        return [[self.activation_fn(np.mean(in_array[i:i + self.pool_size[0], j:j + self.pool_size[1]]))
                 for i in xrange(0, in_array_h, self.pool_size[0])]
                for j in xrange(0, in_array_w, self.pool_size[1])]
