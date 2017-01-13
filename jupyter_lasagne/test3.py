#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年01月11日 星期三 11时50分51秒
# last modified: 

"""
Gen data
"""

import getpass
from neural_fingerprint_theano.code.rdkit_preprocessing import neuralFingerprintUtils
from neural_fingerprint_theano.code.rdkit_preprocessing import genConvMolFeatures
from neural_fingerprint_theano.code.prediction import seqHelper
from rdkit import Chem
import numpy as np

def prepare_data():
    data_dir = '/home/'+getpass.getuser()+'/git_test/neural_fingerprint_theano/data'
    expr_filename = data_dir+'/csv_files/logSolubilityTest.csv'
    fingerprint_filename = data_dir + '/temp/logSolubilityInput_withRDKITidx.pkl'

    smiles_to_measurement, smiles_to_atom_info, smiles_to_bond_info,\
    smiles_to_atom_neighbors, smiles_to_bond_neighbors, smiles_to_atom_mask,\
    smiles_to_rdkit_list, max_atom_len, max_bond_len, num_atom_features, num_bond_features\
        = seqHelper.read_in_data(expr_filename, fingerprint_filename)
    
    dim_dict = {'max_atom_len': max_atom_len,
                'max_bond_len': max_bond_len,
                'num_atom_features': num_atom_features,
                'num_bond_features': num_bond_features}
    
    print dim_dict
    print len(smiles_to_atom_info.keys())
    print smiles_to_atom_info.values()[0].shape
    print smiles_to_measurement.values()[:10]
    print [x.shape for x in smiles_to_atom_info.values()[:10]]
    print [x.shape for x in smiles_to_atom_mask.values()[:10]]


def main():
    prepare_data()



if __name__ == '__main__':
    main()