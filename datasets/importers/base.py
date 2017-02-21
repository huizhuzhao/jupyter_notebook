#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月09日 星期四 16时51分19秒
# last modified: 

import abc

class Importer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load_dataset(self, **kwargs):
        pass
