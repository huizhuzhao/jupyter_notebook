#!/usr/bin/env python
# encoding: utf-8

import numpy 
import numpy as np
import threading
from copy import deepcopy


class Iterator(object):
    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            # the original code is "if N >= current_index + batch_size" 
            # refer to line 509 in url:
            # https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
            if N > current_index + batch_size: 
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0

            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def gen_batch(X, index_array):
    """
    arguments:
        X: list, tuple, numpy.ndarray
        index_array: list of integers, or 1D array of integers
    return:
        batch of X with same type as X
    """
    if not isinstance(X, (list, tuple, numpy.ndarray)):
        raise Exception("Unvalid dtype {0}; ".format(type(X)) +
             "Only <list>, <tuple> and <numpy.ndarray> are valid.")

    batch_X = []
    for idx in index_array:
        batch_X.append(deepcopy(X[idx]))
        
    if isinstance(X, tuple):
        batch_X = tuple(batch_X)
    elif isinstance(X, np.ndarray):
        batch_X = np.asarray(batch_X, dtype=X.dtype)
       
    return batch_X


class SimpleIterator(Iterator):
    """
    A generator for yielding "batch smiles list" with size "batch_size"
    >>> smiles_list = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']
    >>> smiles_it = SimpleIterator(smiles_list, batch_size=4, shuffle=True, seed=123)
    >>> for ii in range(6):
            print smiles_it.next()
    ['s4', 's0', 's7', 's5']
    ['s8', 's3', 's1', 's6']
    ['s9', 's2']
    ['s7', 's9', 's5', 's2']
    ['s8', 's4', 's0', 's3']
    ['s1', 's6']
    
    arguments:
        X: list, tuple, numpy.ndarray
        batch_size: int
        shuffle: True or False; if the idexs of smiles_list will be shuffled at the begining of every epoch
        seed: int or None; if "shuffle" is True, "seed" will be used as seed of the random number generator 
    
    return:
        batch of X with same type as X
    """
    def __init__(self, X, batch_size, shuffle=True, seed=None):
        if not isinstance(X, (list, tuple, np.ndarray)):
            raise Exception("Unvalid dtype {0}; ".format(type(X)) +
                 "Only <list>, <tuple> and <numpy.ndarray> are valid.")
        self.X = X
        super(SimpleIterator, self).__init__(N=len(X), batch_size=batch_size, shuffle=shuffle, seed=seed)
    
    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        
        batch_X = gen_batch(self.X, index_array)
        return batch_X


class ComplexIterator(Iterator):
    """
    generating batches of numpy arrays
    >>> X = np.arange(12).reshape(4, 3)
    >>> y = np.arange(4)
    >>> array_it = ComplexIterator([X, y], batch_size=2, shuffle=True, seed=None)
    >>> for ii in range(4):
            batch_x, batch_y = array_it.next()
            print batch_x.shape, batch_y.shape
    (2, 3) (2,)
    (2, 3) (2,)
    (2, 3) (2,)
    (2, 3) (2,)

    arguments:
        args_list: list of numpy arrays with same length
    
    return: list of batch arrays 
    """
    def __init__(self, args_list, batch_size, shuffle=True, seed=None):
        if not isinstance(args_list, (list, tuple)):
            raise Exception("Unvalid dtype {0}; ".format(type(X)) +
                 "Only <list>, <tuple> are valid.")
        arg_len = len(args_list[0])
        for ii, arg in enumerate(args_list):
            assert len(arg) == arg_len,\
            'Input args in args_list must be same length; ' \
            'length {0} and {1} received'.format(arg_len, len(arg))
            #setattr(self, "arg_"+str(ii), arg)
        self.args_list = args_list
        super(ComplexIterator, self).__init__(N=arg_len, 
                        batch_size=batch_size, shuffle=shuffle, seed=seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
    
        batch_args_list = []
        for arg in self.args_list:
            batch_arg = gen_batch(arg, index_array)
            batch_args_list.append(batch_arg)

        return batch_args_list
