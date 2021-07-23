from typing import List, Tuple, Union
import utils
import numpy as np
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
import sys
import os

data_dir = '[your_data_dir]'
numpy_dir = '[your_save_dir]'
try:
	with open(data_dir + key, 'rb') as f:
		m1, m2 = pickle.load(f)
	# prepare ligand
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
	# pIC50 to class
	Y = 1 if '[declare positive criteria here]' in key else 0
	sample = {
		'H1': H1,
		'H2': H2,
		'A1': adj1,
		'A2': adj2,
		'Y': Y,
		'V': valid,
		'key': key,
			}
	
	np.savez_compressed(numpy_dir+"/"+'[new_key]',features=sample)
except Exception as e:
	print(e)
	continue

