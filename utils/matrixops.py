#!/usr/bin/env python
# encoding: utf-8

from collections import Counter

import numpy as np




def rng_uniform(low=0., high=1., size=(1, ), seed=123):
    '''
    return random uniformly float in semi-open range [low, high), shape=size
    '''
    rng = np.random.RandomState(seed)
    res = rng.uniform(low, high, size)
    return res

def rng_integers(low=0, high=100, size=(1, ), seed=123):
    '''
    return random uniformly integers in semi-open range [low, high), shape=size
    '''
    rng = np.random.RandomState(seed)
    res = rng.randint(low, high, size)
    return res

def rng_binomial(n=1, p=0.5, size=(1, ), seed=123):
    '''
    return sample numbers in binomial distribution (tests number=n, propability=p, shape=size)
    '''
    rng = np.random.RandomState(seed)
    res = rng.binomial(n, p, size)
    return res

def rng_normal(mu=0., sigma=1., size=(1, ), seed=123):
    '''
    return random floats in normal distribution with params: mu, sigma, size
    '''
    rng = np.random.RandomState(seed)
    res = rng.normal(loc=mu, scale=sigma, size=size)
    return res


def iterate_minibatches(batchsize, inputs, targets, *args, **kwargs):
    """
    inputs: training X matrix
    targets: targets y matrix
    batchsize: integer
    *args: sample_weights matrix etc. len(inputs) == len(targets) == len(arg)
    **kwargs: shuffle=True or shuffle=Flase (default when ignored)

    return: one mini_batch of (X_mb, y_mb, arg_mb)
    >>> inputs, targets = trn_X, trn_y
    >>> sample_weights = weights_mat
    >>> batchsize = 10
    >>> for X_mb, y_mb, weight_mb in iterate_minibatches(
    ...        batchsize, inputs, targets, sample_weights, shuffle=True):
    ...    print X_mb.shape, y_mb.shape, weight_mb.shape
    >>> (10, 784), (10, 10), (10, 1)

    """

    tools.assert_array_list_length(inputs, targets, *args)
    n_samples = len(inputs)

    shuffle = kwargs.get('shuffle', False)

    index = list(range(n_samples))
    if shuffle:
        np.random.shuffle(index)

    num_batches = int(n_samples / batchsize)
    if n_samples % batchsize:
        num_batches += 1

    if isinstance(inputs, np.ndarray):
        for ii in range(num_batches):
            excerpt = index[ii*batchsize:(ii+1)*batchsize]
            res = [inputs[excerpt], targets[excerpt]]
            for arg in args:
                res.append(arg[excerpt])
            yield res

    if isinstance(inputs, list):
        for ii in range(num_batches):
            excerpt = index[ii*batchsize:(ii+1)*batchsize]
            res_inputs = [inputs[idx] for idx in excerpt]
            res_targets = [targets[idx] for idx in excerpt]
            res = [res_inputs, res_targets]
            for arg in args:
                res_arg = [arg[idx] for idx in excerpt]
                res.append(res_arg)
            yield res


def one_hot_transformer(x, n):
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


def sample_balance(X, Y, one_hot=True):
    '''
    dependence: numpy, Counter, vectorize_Y
    input: X - 2D np.darray type with shape=(n_sample, n_feature)
    input: Y - 2D np.darray type with shape=(n_sample, n_class)
            one_hot=False: n_class==1, and Y[i][0] indicates the class label of sample X[i]
            one_hot=True: n_class>=2, and Y[i][j] == 1 if sample X[i] belong to class j,
                            otherwise Y[i][j] == 0
    return: X_new with shape=(n_j_max*n_class, n_feature),
            Y_new with shape=(n_j_max*n_class, n_class)

    suppose we have X, Y in which the sample size of class j is n_j,
    and the maximum size is n_j_max (j_max is the class index); this function did a
    randomly sampling of (n_j_max - n_j) samples from class j with replacement to fill class j
    to the same size of n_j_max, then return the new matrices X_new, Y_new. Now, the total size
    is n_j_max * n_class.

    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    >>> Y = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float64)
    >>> X_new, Y_new = sample_balance(X, Y)
    >>> print X
    [[0.  0.]
     [0.  1.]
     [1.  0.]
     [1.  1.]]
    >>> print Y
    [[1.  0.]
     [1.  0.]
     [1.  0.]
     [0.  1.]]
    >>> print X_new
    [[0.  0.]
     [0.  1.]
     [1.  0.]
     [1.  1.]
     [1.  1.]
     [1.  1.]]
    >>> print Y_new
    [[1.  0.]
     [1.  0.]
     [1.  0.]
     [0.  1.]
     [0.  1.]
     [0.  1.]]

    '''
    if one_hot:
        n_class = Y.shape[1]
        Y = np.argmax(Y, axis=1)
    else:
        Y = Y.ravel()
        n_class = len(set(list(Y)))


    class_count = Counter(Y).items()
    class_count = sorted(class_count, key=lambda x:x[1], reverse=True)

    # idex_dict with key==class_label, value=class_idex
    # the number of different class_idex is different now
    idex_dict = {}
    for item in class_count:
        label = item[0]
        idex = np.where(Y == label)
        idex_dict.update({label:idex[0]})
    # idex_sampling_dict with key==class_label, value=class_idex
    # now, different class_idex are balanced after resampling
    max_label, max_count = class_count[0]
    idex_sampling_dict = {}
    for ii in range(n_class):
        tmp_label, tmp_count = class_count[ii]
        tmp_diff = max_count - tmp_count
        tmp_idex_sampling = np.random.choice(idex_dict[tmp_label], size=(tmp_diff, ), replace=True)
        tmp_idex_total = np.hstack((tmp_idex_sampling, idex_dict[tmp_label]))
        idex_sampling_dict.update({tmp_label:tmp_idex_total})

    # X_new, Y_new are re-organized according to class_idex in idex_sampling_dict
    for ii in range(n_class):
        tmp_label = class_count[ii][0]
        tmp_X = X[idex_sampling_dict[tmp_label]]
        tmp_Y = Y[idex_sampling_dict[tmp_label]]
        try:
            X_new = np.vstack((X_new, tmp_X))
            Y_new = np.hstack((Y_new, tmp_Y))
        except Exception:
            X_new = tmp_X
            Y_new = tmp_Y

    if one_hot:
        Y_new = vectorize_Y(Y_new, n_class)
    return X_new, Y_new


