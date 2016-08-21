# a convolution layer with a following pooling layer

import numpy as np
from sigmoid import sigmoid
from sigmoid import sigmoid_d
from flatten_poolingLayer_output import flatten_poolingLayer_output
import conv2d
from FullyConnectedLayer import FullyConnectedLayer


class ConvPoolingLayer(object):
    def __init__(self, input_shape, filter_shape, pool_size=(2, 2), activation_fn=sigmoid, activation_fn_d=sigmoid_d,
                 use_max_pooling=True):
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
        self.activation_fn_d = activation_fn_d

        # number of outputs after pooling
        self.num_output = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size)
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(1.0 / self.num_output), size=filter_shape)
        self.biases = np.random.normal(loc=0.0, scale=1.0, size=(filter_shape[0],))

    # accept input, and return result of forward propagation
    def calculate(self, input_images):
        self.input_images = input_images
        conv_out = conv2d.conv2d(self.input_images, self.weights, self.input_shape)
        # add bias to conv_out
        # for each image in a mini batch
        for mb in xrange(self.input_shape[0]):
            # for each generated feature map
            for k in xrange(self.filter_shape[0] * self.filter_shape[1]):
                conv_out[mb][k] = np.asarray(conv_out[mb][k]) + self.biases[k]
        self.conv_out = conv_out
        # pooling and return
        if self.use_max_pooling:
            self.zs = [
                [np.asarray(self.max_pooling(conv_out[mb][k])) for k in
                 xrange(self.filter_shape[0] * self.filter_shape[1])]
                for mb in xrange(self.input_shape[0])]
        else:
            self.zs = [
                [np.asarray(self.mean_pooling(conv_out[mb][k])) for k in
                 xrange(self.filter_shape[0] * self.filter_shape[1])]
                for mb in xrange(self.input_shape[0])]
        self.out = [[np.asarray(self.activation_fn(self.zs[mb][k])) for k in
                     xrange(self.filter_shape[0] * self.filter_shape[1])]
                    for mb in xrange(self.input_shape[0])]
        return self.out

    def max_pooling(self, in_array):
        in_array_h, in_array_w = in_array.shape
        return [[np.amax(in_array[i:i + self.pool_size[0], j:j + self.pool_size[1]])
                 for i in xrange(0, in_array_h, self.pool_size[0])]
                for j in xrange(0, in_array_w, self.pool_size[1])]

    def mean_pooling(self, in_array):
        in_array_h, in_array_w = in_array.shape
        return [[np.mean(in_array[i:i + self.pool_size[0], j:j + self.pool_size[1]])
                 for i in xrange(0, in_array_h, self.pool_size[0])]
                for j in xrange(0, in_array_w, self.pool_size[1])]

    def BP(self, eta, BP_delta, next_layer, previous_layer_out):
        # return delta of this layer
        # and update weights / biases

        delta = []
        # if the next layer is fully connect layer
        if (next_layer.__class__ == FullyConnectedLayer):
            out_shape = np.asarray(self.zs[0]).shape
            flat_zs = flatten_poolingLayer_output(self.zs)
            delta = [np.dot(w.transpose(), d) * self.activation_fn_d(z) for w, d, z in
                     zip(next_layer.weights, BP_delta, flat_zs)]
            # restore the shape of delta, this is the delta on pooling layer
            delta = [im.reshape(out_shape) for im in delta]
        else:
            print 'Error! Not implemented yet!'

        # pooling layer does'nt need update
        # calculate delta on convolution layer
        if (self.use_max_pooling):
            delta = self.delta_BP_max_pooling(delta)
        else:
            print 'Error! Not implemented yet!'

        # update biases
        nabla_b = np.asarray(delta)
        nabla_b = nabla_b.mean(axis=0)
        nabla_b = nabla_b.mean(axis=1)
        nabla_b = nabla_b.mean(axis=1)
        self.biases = self.biases - eta * nabla_b

        # update weights
        conv_out_shape = self.conv_out[0][0].shape
        nabla_w_batch = []
        # for each image in a mini batch
        for mb in xrange(self.input_shape[0]):
            nabla_w_im = []
            # for each kernel
            for k in xrange(self.filter_shape[0]):
                nabla_w_f_in = []
                # for each input feature map
                for f_in in xrange(self.input_shape[1]):
                    target_output_feature_index = f_in * self.filter_shape[0] + k
                    nabla_w_k = []
                    # for each element in conv_out
                    for i in xrange(conv_out_shape[0]):
                        for j in xrange(conv_out_shape[1]):
                            nabla_w_conv_pnt = self.input_images[mb][f_in][
                                               i:(i + self.filter_shape[2]),
                                               j:(j + self.filter_shape[3])] * \
                                               self.conv_out[mb][target_output_feature_index][i, j]
                            nabla_w_k.append(nabla_w_conv_pnt)
                    nabla_w_k = np.asarray(nabla_w_k).mean(axis=0)
                    nabla_w_f_in.append(nabla_w_k)
                nabla_w_im.append(nabla_w_f_in)
            nabla_w_batch.append(nabla_w_im)
        nabla_w = np.asarray(nabla_w_batch).mean(axis=0)
        # update network parameter
        self.weights = self.weights - eta * nabla_w

    def delta_BP_max_pooling(self, delta_in):
        # calculate delta on convolution when using max_pooling
        delta = np.zeros(np.asarray(self.conv_out).shape)

        conv_out_shape = self.conv_out[0][0].shape
        # for each image in a mini batch
        for mb in xrange(self.input_shape[0]):
            # for each feature map
            for f in xrange(self.filter_shape[0] * self.filter_shape[1]):
                # for each pool
                for i in xrange(0, conv_out_shape[0], self.pool_size[0]):
                    for j in xrange(0, conv_out_shape[1], self.pool_size[1]):
                        d = delta_in[mb][f][i / self.pool_size[0], j / self.pool_size[1]]
                        target_conv_mat = self.conv_out[mb][f][i:(i + self.pool_size[0]),
                                          j:(j + self.pool_size[1])]
                        delta[mb, f, i:(i + self.pool_size[0]), j:(j + self.pool_size[1])] \
                            [np.unravel_index(target_conv_mat.argmax(), target_conv_mat.shape)] = d
        return delta
