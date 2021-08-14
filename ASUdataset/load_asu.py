# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/3/8
"""
import numpy as np
from scipy.io import loadmat

DATASETS = {
    'Arcene': 'load_Arcene',
    'Colon': 'load_Colon',
    'Lung': 'load_Lung',
    'Prostate_GE': 'load_ProGE',
    'TOX_171': 'load_TOX_171',
    'Yale': 'load_Yale',
}


#############basic function################
def load_hdmat(path):
    m = loadmat(path)
    if 'data' in m:
        X, y = m['data'][:, 1:], m['data'][:, 0]
    else:
        X, Y = m['X'], m['Y']
        y = Y.ravel()
    return X, y


def load(dataset):
    if dataset in DATASETS:
        return eval(DATASETS[dataset])()
    else:
        print('dataset {} is invalid'.format(dataset))


#################load_DataSet()#################
def load_Arcene():
    name = 'Arcene'
    path = '../ASUdataset/DataMat/arcene.mat'
    X, y = load_hdmat(path)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Colon():
    name = 'Colon'
    path = '../ASUdataset/DataMat/colon.mat'
    X, y = load_hdmat(path)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Lung():
    name = 'Lung'
    path = '../ASUdataset/DataMat/lung.mat'
    X, y = load_hdmat(path)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_ProGE():
    name = 'Prostate_GE'
    path = '../ASUdataset/DataMat/Prostate_GE.mat'
    X, y = load_hdmat(path)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_TOX_171():
    name = 'TOX_171'
    path = '../ASUdataset/DataMat/TOX_171.mat'
    X, y = load_hdmat(path)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Yale():
    name = 'Yale'
    path = '../ASUdataset/DataMat/Yale.mat'
    X, y = load_hdmat(path)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


if __name__ == '__main__':
    load_Yale()
    load_Lung()
    load_ProGE()
    load_Arcene()

    load_Colon()

    load('Prostate_GE')
