#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年01月23日 星期一 16时18分29秒
# last modified: 

from keras.models import Sequential, Model
from keras.layers import Dense, Input

def main():
    mlp = Sequential()
    dense = Dense(input_shape=(784,), output_dim=64, activation='relu')
    mlp.add(dense)

    print dense.inbound_nodes[0]
    print mlp.inbound_nodes[0] 
    print mlp.outbound_nodes


    """
    print mlp.layers
    print mlp.model
    print mlp.outputs
    print mlp.inbound_nodes
    print mlp.outbound_nodes
    """


if __name__ == '__main__':
    main()
