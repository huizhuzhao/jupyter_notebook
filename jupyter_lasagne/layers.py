#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年01月10日 星期二 11时38分07秒
# last modified: 

from lasagne.layers import InputLayer, DenseLayer, MergeLayer

class FingerprintHiddensLayer(MergeLayer):
    def __init__(self, incoming, input_feature_num, input_atom_num, input_bond_num,
        hidden_units_num, max_atom_len, p_dropout=0.0,\
        W_neighbors=lasagne.init.GlorotUniform(),b_neighbors=lasagne.init.Constant(0.),\
        W_atoms=lasagne.init.GlorotUniform(),b_atoms=lasagne.init.Constant(0.),\
        nonlinearity=nonlinearities.rectify,batch_normalization=True, **kwargs):
        super(FingerprintHiddensLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        #initlialize the values of the weight matrices I'm going to use to transform
        self.W_neighbors = self.add_param(W_neighbors,(input_feature_num+input_bond_num,hidden_units_num),name="W_neighbors")
        self.b_neighbors = self.add_param(b_neighbors,(hidden_units_num,),name="b_neighbors",regularizable=False)

        self.W_atoms = self.add_param(W_atoms,(input_atom_num,hidden_units_num),name="W_atoms")
        self.b_atoms = self.add_param(b_atoms,(hidden_units_num,),name="b_atoms",regularizable=False)

        self.num_units = hidden_units_num
        self.atom_num = input_atom_num
        self.input_feature_num = input_feature_num

        self.p_dropout = p_dropout
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.length = max_atom_len

def buildCNN(input_atom, input_bond, input_atom_index, input_bond_index, input_mask, max_atom_len, max_bond_len):
    dropout_prob = 0.0
    network_vals = {}

    #first encode one set of data
    l_in_atom = InputLayer(shape=(None,max_atom_len,input_atom_dim), input_var=input_atom)
    l_in_bond = InputLayer(shape=(None,max_bond_len,input_bond_dim), input_var=input_bonds)

    l_index_atom = InputLayer(shape=(None,max_atom_len,input_index_dim), input_var=input_atom_index)
    l_index_bond = InputLayer(shape=(None,max_atom_len,input_index_dim), input_var=input_bond_index)

    l_mask = InputLayer(shape=(None,max_atom_len), input_var=input_mask)


def main():
    pass
