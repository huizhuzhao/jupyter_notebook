#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年02月09日 星期四 14时37分18秒
# last modified: 

import numpy as np

def main():
    x = np.array([[0, 2], [1, 1], [2, 0]]).T
    cov = np.cov(x)
    print x
    print cov



if __name__ == '__main__':
    main()
