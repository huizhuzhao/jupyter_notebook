#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月09日 星期四 15时55分35秒
# last modified: 

import numpy as np
import copy

def one_hot_transformer(x, n):
    """
    inputs:
        x: list, tuple, or 1D ndarray type, for example [0, 3, 1, 2]
        n: int, satisfying np.max(x) < n

    return:
        one-hot format data
    """
    if isinstance(x, (tuple, list)):
        x = [int(i) for i in x]
        x = np.asarray(x, dtype=np.int)
    elif isinstance(x, np.ndarray):
        assert len(x.shape) == 1, 'x.shape must be of length 1;'\
                + 'len(x.shape): {0} received.'.format(len(x.shape))
        x = np.asarray(x, dtype=np.int)
    else:
        raise Exception('Input arg must be of type "np.ndarray", \
                "list" or "tuple", {0} received.'.format(type(x)))

    assert np.max(x) < n, 'np.max(x): {0} must be less than n: {1}'.format(np.max(x), n)
    y = np.eye(n)[x]
    return y


def train_valid_split(data, labels, train_rate=0.8, shuffle=True, seed=None):
    """
    DATA_TYPE = [<list>, <tuple>, <dict>, <np.ndarray>]
    inputs:
        data: <list>, elements must be one of the data type in DATA_TYPE
        labels: one of the data type in DATA_TYPE
       
        "labels" and all the elements in "data" must have:
            the same data type
            the same length
            refer function::check_args_type_length for details
            
    """
    if isinstance(data, tuple):
        data = list(data)
    check_args_type_length(*(data + [labels]))

    if isinstance(data, (np.ndarray, list, tuple)):
        idx = range(len(data[0]))
    elif isinstance(data[0], dict):
        idx = data[0].keys()
    if shuffle:
        idx = shuffle_idx(idx, seed)

    num_train = int(len(idx) * train_rate)
    idx_train = idx[:num_train]
    idx_valid = idx[num_train:]

    train_data = []
    valid_data = []
    for x in data:
        train_x = gen_batch(x, idx_train)
        valid_x = gen_batch(x, idx_valid)
        train_data.append(train_x)
        valid_data.append(valid_x)

    train_labels = gen_batch(labels, idx_train)
    valid_labels = gen_batch(labels, idx_valid)

    return train_data, train_labels, valid_data, valid_labels


def shuffle_idx(idx_list, seed=None):
    if seed is None:
        seed = np.random.randint(low=0, high=1e6)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(idx_list)
    return idx


def shuffle_args(*args):
    '''
    shuffle the input arguments with the sample random-state seed
    params: numpy.ndarray type of data and must with SAME LENGTH
    return: list of args after shuffled
    >>> x = np.arange(12).reshape(3, 4)
    >>> y = np.arange(3)
    >>> x_new, y_new = shuffle(x, y)
    x_new; y_new
    [[4, 5, 6, 7],
     [8, 9, 10, 11],
     [0, 1, 2, 3]]
    [1, 2, 0]
    '''
    check_args_type_length(*args)

    new_args = []
    if isinstance(args[0], (np.ndarray, list, tuple)):
        idx = shuffle_idx(range(len(args[0])))

    elif isinstance(args[0], dict):
        idx = shuffle_idx(args[0].keys())

    for arg in args:
       new_arg = gen_batch(arg, idx)
       new_args.append(new_arg)

    return new_args


def gen_batch(X, index_array):
    """
    arguments:
        X: dict, list, tuple, numpy.ndarray
        index_array: list of idx, or keys, or 1D array of integers
    return:
        batch of X with same type as X
    """
    if not isinstance(X, (dict, list, tuple, np.ndarray)):
        raise Exception("Unvalid dtype {0}; ".format(type(X)) +
             "Only <dict>, <list>, <tuple> and <numpy.ndarray> are valid.")

    if isinstance(X, dict):
        batch_X = {}
        for idx in index_array:
            batch_X.update({idx: X[idx]})
    else:
        batch_X = []
        for idx in index_array:
            batch_X.append(copy.deepcopy(X[idx]))
        if isinstance(X, tuple):
            batch_X = tuple(batch_X)
        elif isinstance(X, np.ndarray):
            batch_X = np.asarray(batch_X, dtype=X.dtype)
       
    return batch_X


def check_args_type_length(*args):
    data_type = type(args[0])
    data_len = len(args[0])
    for arg in args:
        assert type(arg) == data_type, \
                'Input args must be same type; '\
                'types {0} and {1} received.'.format(data_type, type(arg))
        assert len(arg) == data_len, \
                'Input args must be same length; '\
                'lengths {0} and {1} received.'.format(data_len, len(arg))
    if isinstance(args[0], dict):
        keys = args[0].keys()
        for arg in args:
            assert sorted(keys) == sorted(arg.keys()), \
                    'Input dicts must have same keys.'
    return True
