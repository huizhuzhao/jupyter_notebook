#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月17日 星期五 18时20分29秒
# last modified: 

import theano
import theano.tensor as T

def categorical_crossentropy(y_pred, y_true):
    """
    inputs
    ------
    y_pred: 2D tensor in (0, 1), such as softmax output
    y_true: 2D tensor in [0, 1], including one-hot type

    return
    ------
    cross entropy between an approximating distribution (y_pred) and a true distribution (y_true). 
    """
    return T.nnet.nnet.categorical_crossentropy(y_pred, y_true)

def sparse_categorical_crossentropy(y_pred, y_true):
    """
    same function as "categorical_crossentropy"
    inputs
    ------
    y_pred: 2D tensor in (0, 1), such as softmax output
    y_true: 1D tensor in {0, 1, .., num_cls-1}, class index per data point

    """
    return T.nnet.nnet.categorical_crossentropy(y_pred, y_true)
