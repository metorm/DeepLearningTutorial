# convert the output of a pooling layer to suitable output for fully connected layer
import numpy as np


def flatten_poolingLayer_output(input_image):
    """ input_image: a batch of output obtained from a pooling layer
                    1st dimension: batch
                    2ed dimension: feature map
                    3rd & 4th dimension: 2d numpy array, image height and width
        out: flatten the 3rd & 4th dimension to one single dimension
            1st and 2ed dimension left unmoved
    """
    return [np.asarray(im).flatten() for im in input_image]