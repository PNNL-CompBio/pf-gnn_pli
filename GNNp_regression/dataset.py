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


random.seed(0)


def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    H1 = []
    f_bonds = []
    n_bonds = 0
    if is_ligand:
        for i in range(n):
            features_ligand, size_ligand = utils.atom_feature_ligand(m, i, None, None)
            H1.append(features_ligand)
        H = np.array(H1)
    else:
        for i in range(n):
            features_protein, size_protein = utils.atom_feature_protein(m, i, None, None)
            H.append(features_protein)
        H = np.array(H)
    return H


class MolDataset(Dataset):

    def __init__(self, keys, data_dir):
        self.keys = keys
        self.data_dir = data_dir
        self.n_bonds = 0  # number of bonds
        self.fH1_hold = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # idx = 0
        key = self.keys[idx]
        sample = np.load(self.data_dir +"/"+ key + ".npz", allow_pickle=True)['features']
        sample = sample.item()

        return sample

class DTISampler(Sampler):

    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights) / np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        # return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights)
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    max_natoms_protein = max([len(item['H2']) for item in batch if item is not None])
    max_nbonds_ligand = max([len(item['H1']) for item in batch if item is not None])
    max_natoms_ligand = max([len(item['A1']) for item in batch if item is not None])

    natom = max_natoms_ligand + max_natoms_protein
    H1_feature_size = batch[0]['H1'].shape[1]
    H2_feature_size = batch[0]['H2'].shape[1]
    H1 = np.zeros((len(batch), max_nbonds_ligand, H1_feature_size))  # ligand
    H2 = np.zeros((len(batch), max_natoms_protein, H2_feature_size))  # protein
    A1 = np.zeros((len(batch), max_natoms_ligand, max_natoms_ligand))
    A2 = np.zeros((len(batch), max_natoms_protein, max_natoms_protein))
    Y = np.zeros((len(batch),))
    V = np.zeros((len(batch), natom))
    keys = []

    for i in range(len(batch)):
        nbonds1 = len(batch[i]['H1'])
        natom1 = len(batch[i]['A1'])
        natom2 = len(batch[i]['H2'])
        natom = natom1 + natom2
        H1[i, :nbonds1] = batch[i]['H1']
        H2[i, :natom2] = batch[i]['H2']
        A1[i, :natom1, :natom1] = batch[i]['A1']
        A2[i, :natom2, :natom2] = batch[i]['A2']
        Y[i] = batch[i]['Y']
        V[i, :natom] = batch[i]['V']
        keys.append(batch[i]['key'])

    H1 = torch.from_numpy(H1).float()
    H2 = torch.from_numpy(H2).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    Y = torch.from_numpy(Y).float()
    V = torch.from_numpy(V).float()
    return H1, H2, A1, A2, Y, V, keys







