DataFile = 'data/mnist.pkl.gz'
InputLayerSize = 784 # Depend on data in DataFile
OutputLayerSize = 10 #Depend on data in DataFile. 10 numbers.

import mnist_loader
TrainData,ValidationData,TestData = mnist_loader.load_data(InputLayerSize,OutputLayerSize,DataFile)
print('Data loaded.')
