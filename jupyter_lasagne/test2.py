#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年01月11日 星期三 10时23分02秒
# last modified:
"""
test2
"""
import theano.tensor as T
from lasagne.layers import InputLayer

from layers import FingerprintHiddensLayer

def main():
    #define my theano variables
    input_atom       = T.ftensor3('input_atom')
    input_atom_index = T.itensor3('input_atom_index')
    input_bond      = T.ftensor3('input_bonds')
    input_bond_index = T.itensor3('input_mask_attn')
    input_mask       = T.fmatrix('input_mask_attn')
    target_vals      = T.fvector('output_data')
    
    max_atom_len = 56
    max_bond_len = 63
    num_atom_features = 68
    num_bond_features = 6
    input_index_dim = 6
    fingerprint_dim = 265
    output_dim = 1
    input_atom_dim = num_atom_features
    input_bond_dim = num_bond_features

    l_in_atom = InputLayer(shape=(None, max_atom_len, input_atom_dim),
        input_val=input_atom)
    l_in_bond = InputLayer(shape=(None, max_bond_len, input_bond_dim),
        input_var=input_bond)
    l_index_atom = InputLayer(shape=(None, max_atom_len, input_index_dim),
        input_var=input_atom_index)
    l_index_bond = InputLayer(shape=(None, max_atom_len, input_index_dim),
        input_var=input_bond_index)
    
    l_mask = InputLayer(shape=(None, max_atom_len), input_var=input_mask)
    
    incomings = [l_index_atom, l_index_bond, l_in_atom, l_in_bond, l_in_atom, l_mask]
    hidden_layer = FingerprintHiddensLayer(
        incomings, num_atom_features, num_atom_features,
        num_bond_features, 400, max_atom_len)
    
    print [x.output_shape for x in incomings]
    print l_in_atom.output_shape, l_index_atom.output_shape
    print l_in_bond.output_shape, l_index_bond.output_shape
    print l_mask.output_shape
    print hidden_layer.W_neighbors.shape

    #hidden_layer = FingerprintHiddensLayer

if __name__ == '__main__':
    main()