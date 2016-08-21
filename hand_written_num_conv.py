# convolution - polling version of the same question

import random

import numpy as np

# load data
DataFile = 'data/mnist.pkl.gz'
InputLayerSize = 784  # Depend on data in DataFile
InputW = 28
InputH = 28
OutputLayerSize = 10  # Depend on data in DataFile. 10 numbers.

import mnist_loader

TrainData, ValidationData, TestData = mnist_loader.load_data(InputLayerSize, OutputLayerSize, DataFile)
print('Data loaded.')

# super parameters
mini_batch_size = 30
train_round = 10
eta = 0.01

# create layers
from ConvPoolingLayer import ConvPoolingLayer
from FullyConnectedLayer import FullyConnectedLayer
from SoftmaxLayer import  SoftmaxLayer
from flatten_poolingLayer_output import flatten_poolingLayer_output

# kernel shape of 1st CP layer
CP1stFilter_shape = (20, 1, 5, 5)
CPLayer = ConvPoolingLayer(input_shape=(mini_batch_size, 1, InputH, InputW), filter_shape=CP1stFilter_shape)
FLayer = FullyConnectedLayer(n_in= 20*12*12, n_out= 100)
SmLayer = SoftmaxLayer(n_in=100,n_out=10)


# arrange input data
for TR in xrange(train_round):
    random.shuffle(TrainData)
    mini_batches = [TrainData[k:k + mini_batch_size] for k in xrange(0, len(TrainData), mini_batch_size)]
    batch_counter = 0
    for mini_batch in mini_batches:
        batch_counter += 1
        print 'Training round {0}, batch {1}/{2}'.format(TR+1,batch_counter,len(TrainData)/mini_batch_size)

        # forward propagate
        image_batch = [[np.reshape(image, (InputH, InputW))] for image, label in TrainData[0:mini_batch_size] for i in
                       (1,)]
        CPLayerOut = CPLayer.calculate(image_batch)
        FLayerIn = flatten_poolingLayer_output(CPLayerOut)
        FLayerOut = FLayer.calculate(FLayerIn)
        SmLayerOut = SmLayer.calculate(FLayerOut)

        # backward propagate
        RefResult = [np.asarray(label).flatten() for image,label in TrainData[0:mini_batch_size]]

        # train data accuracy
        result = [int(out.argmax() == ref.argmax()) for out,ref in zip(SmLayerOut,RefResult)]
        print 'Accuracy on train data: {0}'.format(np.asarray(result).mean())

        delta = SmLayer.BP(eta, RefResult, FLayerOut)
        delta = FLayer.BP(eta, delta, SmLayer, FLayerIn)
        CPLayer.BP(eta,delta,FLayer,image_batch)
