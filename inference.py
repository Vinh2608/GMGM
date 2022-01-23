from gnn import gnn
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolfiles import SDMolSupplier, MolFromPDBFile
from dataset import get_atom_feature
from datetime import datetime
import numpy as np
import argparse
import torch
import utils
import os

parser = argparse.ArgumentParser()
parser.add_argument("--active_threshold", help="active threshold", type=float, default = 0.5)
parser.add_argument("--interaction_threshold", help="interaction threshold", type=float, default = 0.1)
parser.add_argument("--close_contact", help="close contact threshold", type=float, default = 5.5)
parser.add_argument("--ckpt", help="saved model file", type=str, default = "save/best_origin_opt123_0.879.pt")
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)

factory = utils.ChemicalFeaturesFactory.get_instance()

class InferenceGNN():
    def __init__(self, args) -> None:
        if args.ngpu>0:
            cmd = utils.set_cuda_visible_device(args.ngpu)
            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]

        self.model = gnn(args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = utils.initialize_model(self.model, device, load_save_file=args.ckpt, gpu=args.ngpu>0)

        self.model.eval()

    @classmethod
    def get_feature_by_group(cls, mol):
        if mol == None:
            return {}
            
        feature_by_group = {}
        for f in factory.GetFeaturesForMol(mol):
            feature_by_group[f.GetAtomIds()] = f.GetFamily()
        return feature_by_group

    @classmethod
    def get_feature_dict(cls, mol=None, feature_by_group={}):
        if mol != None:
            feature_by_group = cls.get_feature_dict(mol)

        feature_dict = {}
        for key in feature_by_group:
            for atom_idx in key:
                if atom_idx in feature_dict:
                    feature_dict[atom_idx].append(feature_by_group[key])
                else:
                    feature_dict[atom_idx] = [feature_by_group[key]]

        return feature_dict

    def prepare_single_input(self, m1, m2, m1_feature=None, m2_feature=None):
        #prepare ligand
        n1 = m1.GetNumAtoms()
        c1 = m1.GetConformers()[0]
        d1 = np.array(c1.GetPositions())
        adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
        if m1_feature == None:
            m1_feature = self.get_feature_dict(m1)
        H1 = get_atom_feature(m1, m1_feature, True)

        #prepare protein
        n2 = m2.GetNumAtoms()
        c2 = m2.GetConformers()[0]
        d2 = np.array(c2.GetPositions())
        adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
        if m2_feature == None:
            m2_feature = self.get_feature_dict(m2)
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
    
        H = np.zeros((batch_size, max_natoms, 2*utils.N_atom_features))
        A1 = np.zeros((batch_size, max_natoms, max_natoms))
        A2 = np.zeros((batch_size, max_natoms, max_natoms))
        V = np.zeros((batch_size, max_natoms))
        
        for i in range(batch_size):
            natom = len(batch_input[i]['H'])
            
            H[i,:natom] = batch_input[i]['H']
            A1[i,:natom,:natom] = batch_input[i]['A1']
            A2[i,:natom,:natom] = batch_input[i]['A2']
            V[i,:natom] = batch_input[i]['V']

        H = torch.from_numpy(H).float().to(self.device)
        A1 = torch.from_numpy(A1).float().to(self.device)
        A2 = torch.from_numpy(A2).float().to(self.device)
        V = torch.from_numpy(V).float().to(self.device)

        return H, A1, A2, V

    def prepare_multi_input(self, list_ligands, list_receptors, lf=None, rf=None):
        list_inputs = []
        if lf != None and rf != None:
            for li, re, lif, ref in zip(list_ligands, list_receptors, lf, rf):
                list_inputs.append(self.prepare_single_input(li, re, lif, ref))
        else:
            for li, re in zip(list_ligands, list_receptors):
                list_inputs.append(self.prepare_single_input(li, re))

        return list_inputs

    def predict_label(self, list_ligands, list_receptors, lf=None, rf=None):
        list_inputs = self.prepare_multi_input(list_ligands, list_receptors, lf, rf)
        input_tensors = self.input_to_tensor(list_inputs)
        results = self.model.test_model(input_tensors)
        return results.cpu().detach().numpy()

    def predict_interactions(self, list_ligands, list_receptors, lf=None, rf=None):
        list_inputs = self.prepare_multi_input(list_ligands, list_receptors, lf, rf)
        input_tensors = self.input_to_tensor(list_inputs)
        results = self.model.get_refined_adjs2(input_tensors)
        return results.cpu().detach().numpy()

def get_feature_by_atom(features, atom):
    return dict(filter(lambda x: atom in x[0], features.items()))

def check_basic_condition(interactions, ligand, receptor, lfg, rfg):
    filterd_interactions = []
    list_interactions = list(interactions.keys())

    while len(list_interactions) > 0:
        interaction = list_interactions.pop(0)
        la, ra = interaction
        laf = get_feature_by_atom(lfg, la) # {(a1, a2): feature}
        raf = get_feature_by_atom(rfg, ra) # {(a1, a2): feature}

        if ("Donor" in laf.values() and "Acceptor" in raf.values()) or \
           ("Donor" in raf.values() and "Acceptor" in laf.values()):
           filterd_interactions.append(interaction)

        if "Aromatic" in laf.values() and receptor.GetAtomWithIdx(int(ra)).GetSymbol() == "C":
            for group, feature in laf.items():
                if feature == "Aromatic":
                    for atom in group:
                        # Remove other atom in aromatic
                        if (atom, ra) in list_interactions:
                            list_interactions.remove((atom, ra))

                        filterd_interactions.append((atom, ra))
                    break

        if "Aromatic" in raf.values() and ligand.GetAtomWithIdx(int(la)).GetSymbol() == "C":
            for group, feature in raf.items():
                if feature == "Aromatic":
                    for atom in group:
                        # Remove other atom in aromatic
                        if (la, atom) in list_interactions:
                            list_interactions.remove((la, atom))

                        filterd_interactions.append((la, atom))
                    break

        if "LumpedHydrophobe" in laf.values() and "Aromatic" in raf.values():
            for lgroup, lfeature in laf.items():
                if lfeature == "LumpedHydrophobe":
                    for rgroup, rfeature in raf.items():
                        if rfeature == "Aromatic":
                            for latom in lgroup:
                                for ratom in rgroup:
                                    if (latom, ratom) in list_interactions:
                                        list_interactions.remove((latom, ratom))

                                    filterd_interactions.append((latom, ratom))
                        break

                    break

        if "LumpedHydrophobe" in raf.values() and "Aromatic" in laf.values():
            for lgroup, lfeature in laf.items():
                if lfeature == "Aromatic":
                    for rgroup, rfeature in raf.items():
                        if rfeature == "LumpedHydrophobe":
                            for latom in lgroup:
                                for ratom in rgroup:
                                    if (latom, ratom) in list_interactions:
                                        list_interactions.remove((latom, ratom))

                                    filterd_interactions.append((latom, ratom))
                        break

                    break
    return filterd_interactions

def read_ground_truth(gt_file):
    gt_interactions = []
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
        while len(lines) > 0:
            line = lines.pop(0).split(",")
            if not line:
                continue

            for ra in line[1:]:
                if ra:
                    gt_interactions.append((int(line[0]), int(ra)))
    return gt_interactions        

def cal_prf(gt_interactions, found_interactions):
    TP = list(set(gt_interactions).intersection(set(found_interactions)))
    precision = len(TP) / len(found_interactions)
    recall = len(TP) / len(gt_interactions)
    F1 = 2 * precision * recall / (precision + recall)
    return precision, recall, F1

if __name__ == '__main__':
    # Initialize
    args = parser.parse_args()
    print(args)
    inference_gnn = InferenceGNN(args)
    groundtruth = read_ground_truth("vidok/ground_truth.csv")

    # Load ligand
    ligands_sdf = SDMolSupplier("vidok/Ligand_6lu7.sdf" )
    # ligands_sdf = SDMolSupplier("pdbbind/refined-set/1b40/1b40_ligand.sdf" )
    ligand = ligands_sdf[0]
    print("ligand", ligand != None)
    # print(ligand.GetNumAtoms())
    lfg = InferenceGNN.get_feature_by_group(ligand)
    lf = InferenceGNN.get_feature_dict(feature_by_group=lfg)
    lp = np.array(ligand.GetConformers()[0].GetPositions())
    
    # Load receptor
    receptor = MolFromPDBFile("vidok/Receptor_ViDok.pdb")
    # receptor = MolFromPDBFile("pdbbind/refined-set/1b40/1b40_pocket.pdb")
    print("receptor", receptor != None)
    # print(receptor.GetNumAtoms())
    rfg = InferenceGNN.get_feature_by_group(receptor)
    rf = InferenceGNN.get_feature_dict(feature_by_group=rfg)
    rp = np.array(receptor.GetConformers()[0].GetPositions())
    
    results = inference_gnn.predict_label([ligand], [receptor], [lf], [rf])
    print("result", results[0] > args.active_threshold)

    if results[0] > args.active_threshold:
        interactions = inference_gnn.predict_interactions([ligand], [receptor], [lf], [rf])
        # print("interactions", interactions[0])
        n_ligand_atom = ligand.GetNumAtoms()
        x_coord, y_coord = np.where(interactions[0] >= args.interaction_threshold)

        interaction_dict = {}
        for x, y in zip(x_coord, y_coord):
            if x < n_ligand_atom and y >= n_ligand_atom \
                and np.linalg.norm(lp[x] - rp[y-n_ligand_atom]) < args.close_contact and (x in lf or y-n_ligand_atom in rf):
                interaction_dict[(x, y-n_ligand_atom)] = interactions[0][x][y]                

            if x >= n_ligand_atom and y < n_ligand_atom and (y, x-n_ligand_atom) not in interaction_dict \
                and np.linalg.norm(lp[y] - rp[x-n_ligand_atom]) < args.close_contact and (y in lf or x-n_ligand_atom in rf):
                interaction_dict[(y, x-n_ligand_atom)] = interactions[0][x][y]

        interaction_list = check_basic_condition(interaction_dict, ligand, receptor, lfg, rfg)

        precision, recall, f1_score = cal_prf(groundtruth, interaction_list)
        print("Precision: %.5f\nRecall: %.5f\nF1 Score: %.5f" % (precision, recall, f1_score))

        with open("interactions/%s.csv" % datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), "w", encoding="utf8") as f:
            f.write("ligand_atom,receptor_atom,latom_feature,ratom_feature\n")
            for key in interaction_list:
                laf, raf = "", ""
                if key[0] in lf:
                    laf = str(lf[key[0]])
                if key[1] in rf:
                    raf = str(rf[key[1]])
                f.write("{:d},{:d},{:s},{:s}\n".format(key[0], key[1], laf, raf))
