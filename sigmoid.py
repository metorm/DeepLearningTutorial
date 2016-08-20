import numpy as np


def sigmoid(z):
    # input can be single number or array
    return 1.0/(1.0+np.exp(-z))