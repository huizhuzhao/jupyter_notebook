#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月09日 星期四 15时33分06秒
# last modified: 

import os
import numpy as np
from professor.datasets.utils.data_utils import one_hot_transformer

def mnist(dataset_dir, one_hot=False, dim_ordering=None):
    """
    arguments:
        dataset_dir: directory that storing the dataset

        dim_ordering:
            None: images data will be returned with shape [n_samples, 784]
            'tf': images data will be returned with shape [n_samples, 28, 28, 1]
            'th': images data will be returned with shape [n_samples, 1, 28, 28]
        one_hot:
            False: (default)
            True: trn_labels/test_labels will be converted to one_hot type with n_classes==10
    return:
        trn_images/test_images: 4D np.ndarray        
        trn_labels/test_labels: 1D np.ndarray (one_hot == False)       
                                2D np.ndarray (one_hot == True)       
    reference:
        url=http://yann.lecun.com/exdb/mnist/
    """
    trn_images, trn_labels, test_images, test_labels = mnist_2D(dataset_dir)

    size = (1, 28, 28)
    if dim_ordering is not None:
        if dim_ordering == 'tf':
            trn_images = image_reshape_tf(trn_images, size)
            test_images = image_reshape_tf(test_images, size)

        elif dim_ordering == 'th':
            trn_images = image_reshape_th(trn_images, size)
            test_images = image_reshape_th(test_images, size)
        else:
            raise Exception('unvalid dim_ordering: {0}'.format(dim_ordering))

    if one_hot:
        trn_labels = one_hot_transformer(trn_labels, 10)
        test_labels = one_hot_transformer(test_labels, 10)

    return trn_images, trn_labels, test_images, test_labels


def mnist_2D(data_dir):
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))


        return trX, trY, teX, teY


def image_reshape_th(X, size):
    '''
    arguments:
        X: 2D np.ndarray data, [n_samples, n_features]
        size: (channel, width, height)

    return: 4D np.ndarray data, [n_samples, channel, width, height]
    '''
    assert len(X.shape) == 2
    X = np.reshape(X, tuple([X.shape[0]]+list(size)))
    return X

def image_reshape_tf(X, size):
    X = image_reshape_th(X, size)
    X = np.transpose(X, (0, 2, 3, 1))
    return X
