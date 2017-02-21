#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月17日 星期五 14时55分37秒
# last modified: 

import keras
from keras import backend as K
from keras import metrics

def binary_accuracy(y_pred, y_true):
    """
    inputs
    ------
    y_pred: 1D tensor in (0, 1), such as sigmoidal output
    y_true: 1D tensor in {0, 1}, discrete labels

    returns
    ------
    mean accuracy rate for binary classification problem
    """
    return metrics.binary_accuracy(y_true, y_pred)

def categorical_accuracy(y_pred, y_true):
    """
    inputs
    ------
    y_pred: 2D tensor in (0, 1), such as softmax output
    y_true: 2D tensor in {0, 1}, one-hot type labels

    returns
    -------
    mean accuracy rate for multi-class classification problem
    """
    return metrics.categorical_accuracy(y_true, y_pred)


def sparse_categorical_accuracy(y_pred, y_true):
    """
    inputs
    ------
    y_pred: 2D tensor in (0, 1), such as softmax output
    y_true: 1D tensor in {0, 1, .., num_cls-1}, discrete labels

    returns
    -------
    mean accuracy rate for multi-class classification problem with sparse labels (y_true)
    """
    # different from keras, line 15 at link:
    # https://github.com/fchollet/keras/blob/master/keras/metrics.py
    return K.mean(K.equal(y_true,
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())))
