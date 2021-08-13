from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from typing import List, Tuple, Union
import utils
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
from rdkit import Chem
from rdkit.Chem import PandasTools
import os
import sys
from os import path
import glob
from dataset import *



if __name__ == '__main__':

    mol_dir = sys.argv[1] #path to pickled protein-ligand mol folder
    numpy_dir = sys.argv[2] #path to save numpy feature files
    with open("keys/PDBBind_eba_sample_labels.pkl", 'rb') as pkl_file: #save your labels in pickle format. (complex_name:pic50_value)
        eba_labels = pickle.load(pkl_file)

    for m1_m2_file in glob.glob(mol_dir+"/*"):
        m1_m2 = m1_m2_file.split("/")[-1]
        target = m1_m2
        if target in eba_labels.keys():
            Y = eba_labels[target]
            with open(m1_m2_file, 'rb') as f:
                m1, m2 = pickle.load(f)
                # prepare ligand
            try:
                n1 = m1.GetNumAtoms()
                adj1 = GetAdjacencyMatrix(m1) + np.eye(n1)
                H1 = get_atom_feature(m1, True)
                # prepare protein
                n2 = m2.GetNumAtoms()
                adj2 = GetAdjacencyMatrix(m2) + np.eye(n2)
                H2 = get_atom_feature(m2, False)
                # no aggregation here
                # node indice for aggregation - kept to be used later on in the model
                valid = np.zeros((n1 + n2,))
                valid[:n1] = 1
                sample = {
                    'H1': H1,
                    'H2': H2,
                    'A1': adj1,
                    'A2': adj2,
                    'Y': Y,
                    'V': valid,
                    'key': m1_m2,
                }
                np.savez_compressed(numpy_dir + "/" + m1_m2, features=sample)
            except Exception as e:
                print("Exception occured =====",e)
        else:
            continue





