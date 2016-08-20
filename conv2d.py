# 2d convolution core calculation

import numpy as np
from scipy import signal


def conv2d(input_image, kernels, input_shape):
    # generate input for the next convolution layer, if there is any
    return [[signal.convolve2d(image[mIdx], kernel[mIdx], mode='valid') \
             # for each kernel
             for kernel in kernels \
             # for each feature map in image and kernel
             for mIdx in xrange(input_shape[1])] \
            # for each image in a mini batch
            for image in input_image]
