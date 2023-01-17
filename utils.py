import numpy as np
import torch
from scipy import sparse
from scipy.spatial import distance_matrix
import os.path
import time
import torch.nn as nn
import networkx as nx

#from rdkit.Contrib.SA_Score.sascorer import calculateScore
#from rdkit.Contrib.SA_Score.sascorer
#import deepchem as dc

N_atom_features = 34


def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(10):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output)==1:
            empty.append(str(i))
            
    if len(empty)<ngpus:
        print ('No really empty GPU')
        exit(-1) 
        
    cmd = ','.join(empty)
    return cmd

def initialize_model(model, device, load_save_file=False, gpu=True):
    done_loading = False
    try:
        if load_save_file:
            if gpu:
                model.load_state_dict(torch.load(load_save_file)) 
            else:
                model.load_state_dict(torch.load(load_save_file, map_location=torch.device('cpu'))) 
            done_loading = True
            print('Loaded model from file: ', load_save_file)
    except:
        print('Could not load model from file: ', load_save_file)
        exit(-1)
    
    if not done_loading:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    # if torch.cuda.device_count() > 1:
    #   print("Let's use", torch.cuda.device_count(), "GPUs!")
    #   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #   model = nn.DataParallel(model)
      
    model.to(device)
    return model

def mol_to_nx(mol, node_features, node_coords):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        aid = atom.GetIdx()
        G.add_node(aid, x=node_features[aid], coord=node_coords[aid])
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx())
        
    return G

def merge_graphs(*graphs):
    G = nx.Graph()
    offset = 0
    for i, graph in enumerate(graphs):
        mol, node_features, node_coords = graph
        
        if i == 0:
            valid = np.array([1,0])
        else:
            valid = np.array([0,0])
        
        for atom in mol.GetAtoms():
            aid = atom.GetIdx()
            G.add_node(aid+offset, x=node_features[aid], valid=valid)
            G.add_edge(aid+offset,
                       aid+offset,
                       edge_attr = [1., 0.])
            
        for bond in mol.GetBonds():
            # dist = np.linalg.norm(node_coords[bond.GetBeginAtomIdx()] - node_coords[bond.GetEndAtomIdx()])
            G.add_edge(bond.GetBeginAtomIdx() + offset,
                       bond.GetEndAtomIdx() + offset,
                       edge_attr = [1., 0.])
        
        # G.add_nodes_from([(node+offset, {'x': data['x'], 'valid': valid}) for node, data in graph.nodes(data=True)])
        # G.add_edges_from([(u+offset, v+offset, {'edge_attr': 1}) for u, v, data in graph.edges(data=True)])
        # for u, v, data in graph.edges(data=True):
        #     dist = np.linalg.norm(graph.nodes[u]['coord'] - graph.nodes[v]['coord'])
        #     G.add_edge(u+offset, v+offset, edge_attr = dist)
        
        offset = G.number_of_nodes()
    
    if len(graphs) == 2:
        # Add edge attribute for the edge between the two graphs as the distance between the two graphs
        g1_nnodes = graphs[0][0].GetNumAtoms()
        g2_nnodes = graphs[1][0].GetNumAtoms()
        dist_mat = distance_matrix(graphs[0][2], graphs[1][2])
        close_contacts = np.where(dist_mat < 5)
        
        for u, v in zip(close_contacts[0], close_contacts[1]):
            G.add_edge(u, v+g1_nnodes, edge_attr = [0., dist_mat[u,v]])
            G.nodes[v+g1_nnodes]['valid'][1] = 1
        
    return G

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def construct_atom_feature_vector(m, atom_i, features):

    atom = m.GetAtomWithIdx(atom_i)
    
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic(), int("Donor" in features), int("Acceptor" in features),
                    int("PosIonizable" in features), int("NegIonizable" in features),
                    int("LumpedHydrophobe" in features)])    # (10, 7, 5, 6, 6) --> total 34

    
class ChemicalFeaturesFactory:
    """This is a singleton class for RDKit base features."""
    _instance = None

    @classmethod
    def get_instance(cls):
        try:
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if not cls._instance:
            fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            cls._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)
        return cls._instance