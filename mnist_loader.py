import cPickle
import gzip

import numpy as np

def load_cPickle_data(file):
    f = gzip.open(file)
    TrainData, ValidationData, TestData = cPickle.load(f)
    f.close()
    return (TrainData,ValidationData,TestData)

def build_result_vector(th,size):
    r = np.zeros((size,1))
    r[th] = 1.0
    return r

def load_data(InputLayerSize,OutputLayerSize,file):
    TrainData,ValidationData,TestData = load_cPickle_data(file)
    # Below: _d is data, _l is label (result)
    Tr_d = [np.reshape(d,(InputLayerSize,1)) for d in TrainData[0]]
    Tr_l = [build_result_vector(y,OutputLayerSize) for y in TrainData[1]]
    TrainData = zip(Tr_d,Tr_l)
    Val_d = [np.reshape(d,(InputLayerSize,1)) for d in ValidationData[0]]
    ValidationData = zip(Val_d,ValidationData[1])
    Tst_d = [np.reshape(d,(InputLayerSize,1)) for d in TestData[0]]
    Tst_l = [build_result_vector(y,OutputLayerSize) for y in TestData[1]]
    TestData = zip(Tst_d,Tst_l)
    return (TrainData,ValidationData,TestData)
