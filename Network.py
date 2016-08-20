import numpy as np
import random

class Network(object):

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
        # no need to save intermedia values, because this function is only used for evaluation
        for w,b in zip(self.weights,self.biases):
            a = sigmoid(np.dot(w,a)+b) # matrix multiply
        return a

    def BPCore(self, Input, RefResult):
        # calculate \nabla{b} and \nabla{w} using backward propagation
        # allocate memory for nabla results
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # --------- feed forward ----------
        # first activation array is input
        actv = Input
        # save every intermedia activation arrar
        actvs = [actv]
        # save every intermedia z (start from layer 2)
        zs = []
        # do feed forward
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,actv) + b
            zs.append(z)
            actv = sigmoid(z)
            actvs.append(actv)
        # --------- backward propagate ---------
        # for the last layer
        delta = cross_entropy_cost_derivate(actvs[-1], RefResult)
        # foundation equation 3
        nabla_b[-1] = delta
        # foundation equation 4
        nabla_w[-1] = np.dot(delta, actvs[-2].transpose())
        # for -2 ~ 1 layers
        for l in xrange(2,self.NumOfLayers):
            z = zs[-l]
            zp = sigmoid_d(z)
            # update delta
            delta = np.dot(self.weights[-l + 1].transpose(),delta) * zp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,actvs[-l-1].transpose())
        return (nabla_b,nabla_w)

    def update_network(self, TrainData, eta):
        # update weights and biases of network according a batch of examples in TrainData
        # calculate \nabla{b} and \nabla{w} according to each example
        # then use mean value from all examples in gradient descent
        # eta is the learning rate
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # for each example
        for Input,RefResult in TrainData:
            nabla_b_i,nabla_w_i = self.BPCore(Input,RefResult)
            nabla_b = [nb + nbi for nb,nbi in zip(nabla_b,nabla_b_i)]
            nabla_w = [nw + nwi for nw,nwi in zip(nabla_w,nabla_w_i)]
        # do update
        self.weights = [ow - eta * (nw / len(TrainData)) for ow,nw in zip(self.weights,nabla_w)]
        self.biases = [ob - eta * (nb / len(TrainData)) for ob,nb in zip(self.biases,nabla_b)]

    def evaluate(self,TestData):
        # use FeedForward on each test sample
        r = [(np.argmax(self.FeedForward(T)),y) for (T,y) in TestData]
        # how many of them is right?
        return sum([int(x==y) for (x,y) in r])

    def StochasticGradientDesent(self, TrainData, TrainRound, BatchSize, eta, TestData = None):
        # wrapper function to organise all above functions
        # split TrainData into batches, feed them to update_network in batch-by-batch manner
        # repeat the procedure for TrainRound times
        # after each train round, apply TestData to evaluate the network, and output result
        NumOfTrainData = len(TrainData)
        for i in xrange(TrainRound):
            # rearrange train data randomly
            random.shuffle(TrainData)
            # in python expression array[a:b], b can be larger than len(array)
            # when b > len(array), array[a:b] = array[a:]
            TrainDataBatches = [TrainData[k:k+BatchSize] for k in xrange(0,NumOfTrainData,BatchSize)]
            # update network
            for batch in TrainDataBatches:
                self.update_network(batch,eta)
            # test and output if required
            if TestData:
                print("Round {0}: {1} / {2} correct"
                      "...".format(i+1,self.evaluate(TestData),len(TestData)))
            else:
                print("Round {0} train finished ...".format(i))

# calculate derivate of square cost function
def cross_entropy_cost_derivate(OutputActivation, ReferenceResult):
    return OutputActivation - ReferenceResult

# sigmoid functions
# sigmoid
def sigmoid(z):
    # input can be signle number or array
    return 1.0/(1.0+np.exp(-z))
# devivate of sigmoid
def sigmoid_d(z):
    r = sigmoid(z)
    return r*(1-r)
