from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import utils
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
random.seed(0)

def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    H1 = []
    
    if is_ligand:
        for i in range(n):
            features_ligand, size_ligand = utils.atom_feature_ligand(m, i, None, None)
            H1.append(features_ligand)
        H = np.array(H1)
        H = np.concatenate([H, np.zeros((n,31))], 1)
    else:
         for i in range(n):
             features_protein, size_protein = utils.atom_feature_protein(m, i, None, None)
             H.append(features_protein)
         H = np.array(H)
         H = np.concatenate([np.zeros((n,43)), H], 1)
    return H        

class MolDataset(Dataset):

    def __init__(self, keys, data_dir):
        self.keys = keys
        self.data_dir = data_dir

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = np.load("/qfs/projects/mars_pli/Altman_GNN_Integration/PDBBind_DUDE_GNNdf_8A/pdb_dude_numpyfiles_8A_3_26_2021/"+key+".npz", allow_pickle=True)['features']
        sample = sample.item()

        return sample

class DTISampler(Sampler):

    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self):
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    
    H = np.zeros((len(batch), max_natoms, 74))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    Y = np.zeros((len(batch),))
    V = np.zeros((len(batch), max_natoms))
    keys = []
    
    for i in range(len(batch)):
        natom = len(batch[i]['H'])
        
        H[i,:natom] = batch[i]['H']
        A1[i,:natom,:natom] = batch[i]['A1']
        A2[i,:natom,:natom] = batch[i]['A2']
        Y[i] = batch[i]['Y']
        V[i,:natom] = batch[i]['V']
        keys.append(batch[i]['key'])

    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    Y = torch.from_numpy(Y).float()
    V = torch.from_numpy(V).float()
    return H, A1, A2, Y, V, keys

