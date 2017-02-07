#!/usr/bin/env python
# encoding: utf-8
"""
# author: huizhu
# created time: 2017年01月10日 星期二 11时38分07秒
# last modified:
"""
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.layers import InputLayer, DenseLayer, MergeLayer
from lasagne.nonlinearities import rectify
from lasagne.random import get_rng


class FingerprintHiddensLayer(MergeLayer):
    """
    Hidden Layer
    """
    def __init__(self, incoming, input_feature_num, input_atom_num, input_bond_num,\
            hidden_units_num, max_atom_len, p_dropout=0.0,\
            W_neighbors=lasagne.init.GlorotUniform(), b_neighbors=lasagne.init.Constant(0.),\
            W_atoms=lasagne.init.GlorotUniform(), b_atoms=lasagne.init.Constant(0.),\
            nonlinearity=rectify, batch_normalization=True, **kwargs):
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        super(FingerprintHiddensLayer, self).__init__(incoming, **kwargs)

        #initlialize the values of the weight matrices I'm going to use to transform
        self.W_neighbors = self.add_param(
            W_neighbors, (input_feature_num+input_bond_num, hidden_units_num), name="W_neighbors")
        self.b_neighbors = self.add_param(
            b_neighbors, (hidden_units_num,), name="b_neighbors", regularizable=False)

        self.W_atoms = self.add_param(
            W_atoms, (input_atom_num, hidden_units_num), name="W_atoms")
        self.b_atoms = self.add_param(
            b_atoms, (hidden_units_num,), name="b_atoms", regularizable=False)

        self.num_units = hidden_units_num
        self.atom_num = input_atom_num
        self.input_feature_num = input_feature_num

        self.p_dropout = p_dropout
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.length = max_atom_len

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        atom_mask = inputs[5]
        atom_degree_list = inputs[0]
        bond_degree_list = inputs[1]
        atom_list = inputs[2]
        bond_list = inputs[3]
        # atom_mask.dimshuffle(0, 1, 'x'): (m, n) -> (m, n, 1)
        features_list = inputs[4] * atom_mask.dimshuffle(0, 1, 'x')

        batch_size, mol_length, num_atom_feat = atom_list.shape
        atom_list_reshape = atom_list.reshape((batch_size*mol_length, num_atom_feat))

        atom_transformed_flat = T.dot(atom_list_reshape, self.W_atoms) + self.b_atoms
        atom_transformed = atom_transformed_flat.reshape((batch_size, mol_length, self.num_units))
        


