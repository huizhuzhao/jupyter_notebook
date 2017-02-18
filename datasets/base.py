#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月09日 星期四 09时48分06秒
# last modified:

import numpy as np
from jupyter_notebook.datasets.utils.data_iterator import ComplexIterator
from jupyter_notebook.datasets.utils import data_utils

class Dataset(object):
    """
    a simple data wrapper for numpy array


    # Example:
        >>> dataset_dir = 'dir_where_mnist_dataset_located'
        >>> mnist_importer = MnistImporter(dataset_dir)
        >>> dataset = Dataset()
        >>> dataset.load_importer(mnist_importer)

        now we can use "dataset.data" and "dataset.labels" 
        and geting the "X" and "Y" parts of training data
        noting that "X" is a list of np.ndarrays  
        >>> X, Y = dataset.data, dataset.labels
        >>> X[0].shape, Y.shape
        (60000, 784), (60000, 1)

        >>> dataset.gen_iterator(batch_size=32)
        >>> batch_X, batch_Y = dataset.next() # noting "batch_X" is a list of np.ndarrays
        >>> batch_X[0].shape, batch_Y.shape
        >>> (32, 784), (32, )
        >>> train_dataset, valid_dataset = dataset.train_valid_split(train_rate=0.8)
        >>> train_dataset.num_examples, valid_dataset.num_examples
        >>> 56000 14000
        >>> train_dataset.gen_iterator(64)
        >>> [x.shape for x in train_dataset.next()]
        >>> (64, 784), (64, )

    """

    def __init__(self, data=None, labels=None, seed=None, batch_size=32, shuffle=True):
        """
        DATA_TYPE = [<list>, <tuple>, <dict>, <np.ndarray>]
        inputs:
            data: <list>, elements must be one of the data type in DATA_TYPE
            labels: one of the data type in DATA_TYPE
        "labels" and all the elements in "data" must have:

        """
        self._data = data
        self._labels = labels
        seed = seed if seed is not None else np.random.randint(low=0, high=1e6)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        if data is not None:
            self._num_examples = len(data)
            self.gen_iterator(batch_size, shuffle)
        else:
            self._num_examples = 0

    def load_importer(self, importer):
        """
        inputs:
            importer: Object that can import specific data
        """
        importer.load_dataset()
        data, labels = importer.output()

        self._data = data
        self._labels = labels
        self._num_examples = len(data)
        self.gen_iterator(self.batch_size, self.shuffle)

    def gen_iterator(self, batch_size, shuffle=True):
        seed = self.rng.randint(low=0, high=1e6)
        self._dataset_it = ComplexIterator(self._data + [self._labels],
                                batch_size=batch_size, shuffle=shuffle, seed=seed)

    def train_valid_split(self, train_rate=0.8, shuffle=True):
        """
        inputs:
            train_rate: train dataset rate
            shuffle: if dataset will be shuffled before splitted, True default
            seed: random state seed, if shuffle==True, seed will be used as random state seed
                  for random shuffle
        """
        seed = self.rng.randint(low=0, high=1e6)
        train_data, train_labels, valid_data, valid_labels = data_utils.train_valid_split(
                self._data, self._labels, train_rate, shuffle, seed)

        seed1 = self.rng.randint(low=0, high=1e6)
        seed2 = self.rng.randint(low=0, high=1e6)
        train_dataset = Dataset(train_data, train_labels, 
                    seed=seed1, batch_size=self.batch_size, shuffle=self.shuffle)

        valid_dataset = Dataset(valid_data, valid_labels,
                    seed=seed2, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_dataset, valid_dataset

    def next(self):
        """
        return:
            mini_batch data
        """
        batch_dataset = self._dataset_it.next()
        return batch_dataset[:-1], batch_dataset[-1]

    @property
    def labels(self, **kwargs):
        """
        return:
            self.labels
        """
        return self._labels

    @property
    def data(self, **kwargs):
        """
        return:
            self.data
        """
        return self._data

    @property
    def num_examples(self):
        return self._num_examples

    def stats(self, **kwargs):
        """
        get statistics
        """
        pass
