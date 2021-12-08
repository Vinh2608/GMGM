from gnn import gnn
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolfiles import SDMolSupplier, MolFromPDBFile
from dataset import get_atom_feature
import numpy as np
import argparse
import torch
import utils
import os

parser = argparse.ArgumentParser()
parser.add_argument("--active_threshold", help="active threshold", type=int, default = 0.5)
parser.add_argument("--interaction_threshold", help="interaction threshold", type=int, default = 0.5)
parser.add_argument("--ckpt", help="saved model file", type=str, default = "")
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)

class InferenceGNN():
    def __init__(self, args) -> None:
        if args.ngpu>0:
            cmd = utils.set_cuda_visible_device(args.ngpu)
            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]

        self.model = gnn(args)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = utils.initialize_model(self.model, device, load_save_file=args.ckpt)

        self.model.eval()

    def prepare_single_input(self, m1, m2):
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

        #node indice for aggregation
        valid = np.zeros((n1+n2,))
        valid[:n1] = 1

        sample = {
                  'H':H, \
                  'A1': agg_adj1, \
                  'A2': agg_adj2, \
                  'V': valid, \
                  }

        return sample

    def input_to_tensor(self, batch_input):
        max_natoms = max([len(item['H']) for item in batch_input if item is not None])
        batch_size = len(batch_input)
    
        H = np.zeros((batch_size, max_natoms, 56))
        A1 = np.zeros((batch_size, max_natoms, max_natoms))
        A2 = np.zeros((batch_size, max_natoms, max_natoms))
        V = np.zeros((batch_size, max_natoms))
        
        for i in range(batch_size):
            natom = len(batch_input[i]['H'])
            
            H[i,:natom] = batch_input[i]['H']
            A1[i,:natom,:natom] = batch_input[i]['A1']
            A2[i,:natom,:natom] = batch_input[i]['A2']
            V[i,:natom] = batch_input[i]['V']

        H = torch.from_numpy(H).float()
        A1 = torch.from_numpy(A1).float()
        A2 = torch.from_numpy(A2).float()
        V = torch.from_numpy(V).float()

        return H, A1, A2, V

    def prepare_multi_input(self, list_ligands, list_receptors):
        list_inputs = []
        for li, re in zip(list_ligands, list_receptors):
            list_inputs.append(self.prepare_single_input(li, re))

        return list_inputs

    def predict_label(self, list_ligands, list_receptors):
        list_inputs = self.prepare_multi_input(list_ligands, list_receptors)
        input_tensors = self.input_to_tensor(list_inputs)
        results = self.model.test_model(input_tensors)
        return results.cpu().detach().numpy()

    def predict_interactions(self, list_ligands, list_receptors):
        list_inputs = self.prepare_multi_input(list_ligands, list_receptors)
        input_tensors = self.input_to_tensor(list_inputs)
        results = self.model.get_refined_adjs2(input_tensors)
        return results.cpu().detach().numpy()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    inference_gnn = InferenceGNN(args)

    # Load ligand
    ligands_sdf = SDMolSupplier("dude/egfr_CHEMBL144760/CHEMBL144760.sdf" )
    ligand = ligands_sdf[0]
    print("ligand", ligand != None)
    
    # Load receptor
    receptor = MolFromPDBFile("dude/egfr_CHEMBL144760/egfr.pdb")
    print("receptor", receptor != None)
    
    results = inference_gnn.predict_label([ligand], [receptor])
    print("result", results[0] > args.active_threshold)

    if results[0] > args.active_threshold:
        interactions = inference_gnn.predict_interactions([ligand], [receptor])
        # print("interactions", interactions[0])
        n_ligand_atom = ligand.GetNumAtoms()
        x_coord, y_coord = np.where(interactions[0] > args.interaction_threshold)

        print("interaction: (ligand atom, receptor atom)\n")
        for x, y in zip(x_coord, y_coord):
            if x < n_ligand_atom and y >= n_ligand_atom:
                print("(", x, y-n_ligand_atom, ")")

            if x >= n_ligand_atom and y < n_ligand_atom:
                print("(", y, x-n_ligand_atom, ")")
