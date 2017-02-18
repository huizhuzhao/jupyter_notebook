#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月09日 星期四 17时13分43秒
# last modified: 

import numpy as np
from jupyter_notebook.datasets.importer.base import Importer
from jupyter_notebook.datasets.examples.mnist_dataset import mnist


class MnistImporter(Importer):
    def __init__(self, dataset_dir, one_hot=False, dim_ordering=None):
        self.dataset_dir = dataset_dir
        self.one_hot = one_hot
        self.dim_ordering = dim_ordering

    def load_dataset(self):
        """
        generate <data> and <labels>, both are np.ndarray type
        """

        train_X, train_y, valid_X, valid_y = mnist(self.dataset_dir, one_hot=self.one_hot, 
                                                dim_ordering=self.dim_ordering)
        train_X /= 255.
        valid_X /= 255.

        data = np.concatenate([train_X, valid_X], axis=0)
        labels = np.concatenate([train_y, valid_y], axis=0)

        self.data = [data]
        self.labels = labels

    def output(self):
        """
        return: 
            self.data: np.ndarray
            self.labels: np.ndarray
        """
        return self.data, self.labels
