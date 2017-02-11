#!/usr/bin/env python
# encoding: utf-8
# Created Time: 2017年02月11日 星期六 12时39分47秒

import numpy as np

def correlation_pearson(y_pred, y_true):
    """
    return:
        pearson correlation function:
    """
    y_pred, y_true = regression_shape_check(y_pred, y_true)

    cov = np.mean(y_pred * y_true) \
          - np.mean(y_pred) * np.mean(y_true)
    y_pred_std = np.std(y_pred)
    y_true_std = np.std(y_true)
    corr_pearson = cov / (y_pred_std * y_true_std)

    return corr_pearson

def regression_shape_check(y_pred, y_true):
    """
    Assert that "y_pred.shape == y_true.shape" and be the form 
        "(batch_size, )" or "(batch_size, 1)". If inputs shapes are the latter case,
        they will be reshaped to the former one, i.e. "(batch_size, )" and returned.
    """
    assert y_pred.shape == y_true.shape, \
            'Input shapes: {0}, {1} do not match.'.format(y_pred.shape, y_true.shape)

    if len(y_pred.shape) == 1 or len(y_pred.shape) == 2:
        if len(y_pred.shape) == 2:
            assert y_pred.shape[1] == 1, \
                'Input shapes must be (batch_size, ) or (batch_size, 1); {0} received.'.format(
                    y_pred.shape)
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
    else:
        raise Exception("Input shapes must be (batch_size, ) or (batch_size, 1);" +
                "{0} received.".format(y_pred.shape))
    
    return y_pred, y_true

if __name__ == '__main__':
	x = np.random.randn(10)
	y = np.random.randn(10)
	r = correlation_pearson(x, y)
	print r
