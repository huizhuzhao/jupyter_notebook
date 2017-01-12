#!/usr/bin/env python
# encoding: utf-8
# Created Time: 2017年01月12日 星期四 23时05分20秒

import numpy as np
from numpy.matlib import repmat
from jupyter_notebook.utils import data_iterator
from keras.models import Sequential
from keras.layers import Dense

def gen_data():
    x = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float)
    y = np.asarray([1., 0., 0., 1.], dtype=np.float)
    x = repmat(x, 100, 1)
    y = repmat(y, 100, 1)
    return x, y.flatten()


def build_model():
    model = Sequential()
    model.add(Dense(output_dim=2, input_dim=2, activation='tanh'))
    model.add(Dense(output_dim=2, activation='sigmoid'))
    return model

def train_model(model, data_it, num_batches=10, loop=10):
    loss_metrics = []
    for ii in range(loop):
        tmp_loss_metrics = []
        for jj in range(num_batches):
            batch_x, batch_y = data_it.next()
            batch_loss_metrics = model.train_on_batch(batch_x, batch_y)
            tmp_loss_metrics.append(batch_loss_metrics)
        tmp_loss_metrics = np.mean(np.asarray(tmp_loss_metrics), axis=0)
        loss_metrics.append(tmp_loss_metrics)
    return loss_metrics

def main():
    x, y = gen_data()
    model = build_model()
    print 'data:: x.shape: {0}, y.shape: {1}'.format(x.shape, y.shape)
    print 'model:: input_shape: {0}, output_shape: {1}'.format(
        model.input_shape, model.output_shape)
    
    num_batches = 50
    batch_size = 32
    loop = 100
    data_it = data_iterator.ComplexIterator([x, y], batch_size, shuffle=True, seed=123)
    model_0 = build_model()
    model_0.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model_1 = build_model()
    model_1.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    model_2 = build_model()
    model_2.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    loss_metrics_0 = train_model(model_0, data_it, loop=loop)
    loss_metrics_1 = train_model(model_1, data_it, loop=loop)
    loss_metrics_2 = train_model(model_2, data_it, loop=loop)

    import matplotlib.pyplot as plt
    loss_0 = [m[0] for m in loss_metrics_0]
    metrics_0 = [m[1] for m in loss_metrics_0]
    loss_1 = [m[0] for m in loss_metrics_1]
    metrics_1 = [m[1] for m in loss_metrics_1]
    loss_2 = [m[0] for m in loss_metrics_2]
    metrics_2 = [m[1] for m in loss_metrics_2]
    plt.figure()
    plt.subplot(211)
    plt.plot(loss_0, 'b', label='sgd')
    plt.plot(loss_1, 'g', label='adagrad')
    plt.plot(loss_2, 'r', label='rmsprop')
    plt.legend()
    plt.ylabel('loss')
    plt.subplot(212)
    plt.plot(metrics_0, 'b')
    plt.plot(metrics_1, 'g')
    plt.plot(metrics_2, 'r')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    main()
