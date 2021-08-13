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

    numpy_dir = 'data_numpy_pic50' #path to save numpy feature files
    with open("keys/PDBBind_pIC50_sample_labels.pkl", 'rb') as pkl_file: #save your labels in pickle format. (complex_name:pic50_value)
        pic50_labels = pickle.load(pkl_file)

    for m1_m2_file in glob.glob("data_mol_pkl/*"):
        m1_m2 = m1_m2_file.split("/")[-1]
        target = m1_m2
        if target in pic50_labels.keys():
            Y = pic50_labels[target]
            if float(Y) < 2:
                Y = 2
            if float(Y) > 11:
                Y = 11
            with open(m1_m2_file, 'rb') as f:
                m1, m2 = pickle.load(f)
                # prepare ligand
            try:
                n1 = m1.GetNumAtoms()
                c1 = m1.GetConformers()[0]
                d1 = np.array(c1.GetPositions())
                adj1 = GetAdjacencyMatrix(m1) + np.eye(n1)
                H1 = get_atom_feature(m1, True)

                # prepare protein
                n2 = m2.GetNumAtoms()
                c2 = m2.GetConformers()[0]
                d2 = np.array(c2.GetPositions())
                adj2 = GetAdjacencyMatrix(m2) + np.eye(n2)
                H2 = get_atom_feature(m2, False)

                # aggregation
                H = np.concatenate([H1, H2], 0)
                agg_adj1 = np.zeros((n1 + n2, n1 + n2))
                agg_adj1[:n1, :n1] = adj1
                agg_adj1[n1:, n1:] = adj2
                agg_adj2 = np.copy(agg_adj1)
                dm = distance_matrix(d1, d2)
                agg_adj2[:n1, n1:] = np.copy(dm)
                agg_adj2[n1:, :n1] = np.copy(np.transpose(dm))

                # node indice for aggregation
                valid = np.zeros((n1 + n2,))
                valid[:n1] = 1

                # pIC50 to class

                # if n1+n2 > 300 : return None
                sample = {
                    'H': H,
                    'A1': agg_adj1,
                    'A2': agg_adj2,
                    'Y': Y,
                    'V': valid,
                    'key': m1_m2,
                    }
                np.savez_compressed(numpy_dir + "/" + m1_m2, features=sample)

            except Exception as e:
                print("Exception occured =====",e)
        else:
            continue





