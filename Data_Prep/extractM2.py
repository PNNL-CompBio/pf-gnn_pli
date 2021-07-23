from Bio.PDB import *
from Bio.PDB.PDBIO import Select
from Bio import BiopythonWarning
import warnings
import os
from rdkit import Chem
import rdkit
import numpy as np
from scipy.spatial import distance_matrix

with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)

def extract(ligand, pdb):
    parser = PDBParser()
    if not os.path.exists(pdb):
        return None
    structure = parser.get_structure('protein', pdb)
    ligand_positions = ligand.GetConformer().GetPositions()


    class GlySelect(Select):
        def accept_residue(self, residue):
            residue_positions = np.array([np.array(list(atom.get_vector())) \
                for atom in residue.get_atoms() if 'H' not in atom.get_id()])
                    
            if residue_positions.size != 0 and ligand_positions.size != 0:
                min_dis = np.min(distance_matrix(residue_positions, ligand_positions))
                if min_dis < 8:
                    return 1
                else:
                    return 0
            else:
                return 0    

    io = PDBIO()
    io.set_structure(structure)

    fn = 'BS_tmp_'+str(os.getpid())+'.pdb'
    io.save(fn, GlySelect())
    m2 = Chem.MolFromPDBFile(fn)
    
    return m2



