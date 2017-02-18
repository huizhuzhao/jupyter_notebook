#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月10日 星期五 18时12分03秒
# last modified: 

import numpy as np
import progressbar
import time

def train_model_keras(model, train_dataset, valid_dataset=None, nb_epoch=10, nb_batches=None, progressbar_ON=False):
    """
    inputs
    -------
    model: keras Sequential model
    train_dataset: Object of <Dataset> instance
    valid_dataset: Object of <Dataset> instance or None
    nb_epoch: int
    nb_batches: int or None
    progressbar_ON: True or False

    return
    ------
    loss_metrics_dict:
        keys::train_loss_metrics: list of 1D np.ndarrays
        keys::valid_loss_metrics: list of 1D np.ndarrays or [](if valid_dataset is None)
    """

    def loss_metrics(model, dataset):
        """
        compute loss value and metrics
        """
        if dataset is None:
            return []
        batch_lm = model.test_on_batch(dataset.data, dataset.labels)
        return np.asarray(batch_lm)

    def train_one_epoch(model, dataset, nb_batches, progressbar_ON):
        if progressbar_ON:
            pg_bar = progressbar.ProgressBar(maxval=nb_batches).start()
            time1 = time.time()

        for jj in range(nb_batches):
            batch_x, batch_y = dataset.next()
            model.train_on_batch(batch_x, batch_y)
            if progressbar_ON:
                pg_bar.update(jj+1)

        if progressbar_ON:
            pg_bar.finish()
            time2 = time.time()
            print 'time: {0}'.format(round(time2 - time1, 2))

    train_loss_metrics = []
    valid_loss_metrics = []

    if nb_batches is None:
        train_nb_batches = train_dataset._dataset_it.nb_batches
    else:
        train_nb_batches = nb_batches


    train_lm = loss_metrics(model, train_dataset)
    valid_lm = loss_metrics(model, valid_dataset)
    print '{0}/{1}, train: {2}, valid: {3}'.format(0, nb_epoch, train_lm, valid_lm)
    for ii in range(nb_epoch):
        train_one_epoch(model, train_dataset, train_nb_batches, progressbar_ON)
        train_lm = loss_metrics(model, train_dataset)
        valid_lm = loss_metrics(model, valid_dataset)
        train_loss_metrics.append(train_lm)
        valid_loss_metrics.append(valid_lm)
        print '{0}/{1}, train: {2}, valid: {3}'.format(ii+1, nb_epoch, train_lm, valid_lm)



    return {'train_loss_metrics': train_loss_metrics, 'valid_loss_metrics': valid_loss_metrics}
