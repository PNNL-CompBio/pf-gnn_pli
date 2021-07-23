import utils
import numpy as np
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
import os
import os.path

data_dir = '[your_data_dir]'
numpy_dir = '[your_save_dir]'
try:
	with open(data_dir + key, 'rb') as f:
		m1, m2 = pickle.load(f)

	#prepare ligand
	n1 = m1.GetNumAtoms()
	c1 = m1.GetConformers()[0]
	d1 = np.array(c1.GetPositions())
	adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
	H1 = get_atom_feature(m1, True)

	#prepare protein
	n2 = m2.GetNumAtoms()
	c2 = m2.GetConformers()[0]
	d2 = np.array(c2.GetPositions())
	adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
	H2 = get_atom_feature(m2, False)

	#aggregation
	H = np.concatenate([H1, H2], 0)
	agg_adj1 = np.zeros((n1+n2, n1+n2))
	agg_adj1[:n1, :n1] = adj1
	agg_adj1[n1:, n1:] = adj2
	agg_adj2 = np.copy(agg_adj1)
	dm = distance_matrix(d1,d2)
	agg_adj2[:n1,n1:] = np.copy(dm)
	agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

	Y = 1 if '[declare positive criteria here]' in key else 0

	#node indice for aggregation
	valid = np.zeros((n1+n2,))
	valid[:n1] = 1

	sample = {
		'H':H, \
		'A1': agg_adj1, \
		'A2': agg_adj2, \
		'Y': Y, \
		'V': valid, \
		'key': key, 
			}

	np.savez_compressed(numpy_dir+"/"+'[new_key]',features=sample)
except Exception as e:
	print(e)
        continue
