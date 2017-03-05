#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月09日 星期四 14时37分18秒
# last modified: 

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from jupyter_notebook.datasets.utils.data_utils import one_hot_transformer
from jupyter_notebook.datasets.base import Dataset
from jupyter_notebook.models.utils.training import train_model_keras

class QuadraticImporter():
    def __init__(self):
        pass
    
    def load_dataset(self):
        x = np.linspace(0, 1, 100)
        y1 = (2*(x-0.5))**2
        y2 = (2*(x-0.5))**2 + 0.5
    
        X_1 = np.c_[x, y1]    
        X_2 = np.c_[x, y2]
        y_1 = one_hot_transformer(np.ones_like(x), 2)
        y_2 = one_hot_transformer(np.zeros_like(x), 2)

        data = np.concatenate([X_1, X_2], axis=0)
        labels = np.concatenate([y_1, y_2], axis=0)
        self.data = [data]
        self.labels = labels
    
    def output(self):
        return self.data, self.labels

def build_model():
    model = Sequential()
    model.add(Dense(output_dim=2, input_shape=(2, ), activation='relu', W_regularizer=l2(0.0001))) # changing output_dim, you test different model
    #model.add(Dense(output_dim=3, activation='relu'))
    model.add(Dense(output_dim=2, activation='softmax', W_regularizer=l2(0.0)))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def visualize_model(model, x_min, x_max, y_min, y_max, h, X, y):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = np.reshape(zz.argmax(axis=1), xx.shape)
    
    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y.argmax(axis=1), cmap=plt.cm.coolwarm)
    plt.show()

def main():
    data_importer = QuadraticImporter()
    dataset = Dataset()
    dataset.load_importer(data_importer)
    X, y = dataset.data[0], dataset.labels

    #plt.plot(X[:100, 0], X[:100, 1])
    #plt.plot(X[100:, 0], X[100:, 1], 'r')
    #plt.show()
    model = build_model()
    print X.shape, y.shape
    print model.input_shape, model.output_shape
    print [x.output_shape for x in model.layers]

    nb_batches=150; nb_epoch=70
    batch_x, batch_y = dataset.next()
    print [x.shape for x in dataset.data]
    loss_metrics_dict = train_model_keras(model, train_dataset=dataset, nb_epoch=nb_epoch, nb_batches=nb_batches)

    loss_metrics = loss_metrics_dict['train_loss_metrics']
    loss_list = [x[0] for x in loss_metrics]
    metrics_list = [x[1] for x in loss_metrics]
    #plt.plot(loss_list)
    #plt.plot(metrics_list, 'r')
    plt.show()
    visualize_model(model, 0, 1, 0, 2, 0.01, X, y)



if __name__ == '__main__':
    main()
