from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .utils import N_atom_features, construct_atom_feature_vector
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
import os
from tqdm import tqdm

random.seed(0)

def get_atom_feature(m, feature, is_ligand=True):
    n = m.GetNumAtoms()

    H = []
    atom_feature = []
    for i in range(n):
        if i in feature:
            atom_feature = feature[i]
        else:
            atom_feature = []

        H.append(construct_atom_feature_vector(m, i, atom_feature))

    H = np.array(H)        
    if is_ligand:
        H = np.concatenate([H, np.zeros((n,N_atom_features))], 1)
    else:
        H = np.concatenate([np.zeros((n,N_atom_features)), H], 1)

    return H        

class MolDataset(Dataset):

    def __init__(self, keys, data_dir, train=True):
        self.keys = keys
        self.data_dir = data_dir

        # del_keys = []
        # for i, key in tqdm(enumerate(self.keys)):
        #     with open(os.path.join(self.data_dir, key), 'rb') as f:
        #         m1, m2, _, __ = pickle.load(f)
        #         if m1 == None or m2 == None:
        #             del_keys.append(i)

        # for i_key in del_keys[::-1]:
        #     del self.keys[i_key]

        # if train:
        #     pickle.dump(self.keys, open("train_filtered.pkl", "wb"))
        # else:
        #     pickle.dump(self.keys, open("test_filtered.pkl", "wb"))

        # print("Total key usable:", len(self.keys))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #idx = 0
        key = self.keys[idx]
        with open(os.path.join(self.data_dir, key), 'rb') as f:
            m1, m2, m1_feature, m2_feature = pickle.load(f)

        #prepare ligand
        n1 = m1.GetNumAtoms()
        c1 = m1.GetConformers()[0]
        d1 = np.array(c1.GetPositions())
        adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
        H1 = get_atom_feature(m1, m1_feature, True)

        #prepare protein
        n2 = m2.GetNumAtoms()
        c2 = m2.GetConformers()[0]
        d2 = np.array(c2.GetPositions())
        adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
        H2 = get_atom_feature(m2, m2_feature, False)
        
        #aggregation
        H = np.concatenate([H1, H2], 0)
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
        dm = distance_matrix(d1,d2)
        agg_adj2[:n1,n1:] = np.copy(dm)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

        #node indice for aggregation
        valid = np.zeros((2, n1+n2,))
        valid[0,:n1] = 1
        valid[1,np.unique(np.where(dm < 5)[1])] = 1
        
        #pIC50 to class
        Y = 1 if 'CHEMBL' in key else 0
        Y = float(key.split('_')[-1])

        #if n1+n2 > 300 : return None
        sample = {
                  'H':H, \
                  'A1': agg_adj1, \
                  'A2': agg_adj2, \
                  'Y': Y, \
                  'V': valid, \
                  'key': key, \
                  }

        return sample

class DTISampler(Sampler):

    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self):
        #return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    
    H = np.zeros((len(batch), max_natoms, 2*N_atom_features))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    Y = np.zeros((len(batch),))
    V = np.zeros((len(batch), 2, max_natoms))
    keys = []
    
    for i in range(len(batch)):
        natom = len(batch[i]['H'])
        
        H[i,:natom] = batch[i]['H']
        A1[i,:natom,:natom] = batch[i]['A1']
        A2[i,:natom,:natom] = batch[i]['A2']
        Y[i] = batch[i]['Y']
        V[i,:,:natom] = batch[i]['V']
        keys.append(batch[i]['key'])

    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    Y = torch.from_numpy(Y).float()
    V = torch.from_numpy(V).float()
    
    return H, A1, A2, Y, V, keys

