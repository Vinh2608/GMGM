# %%
import os
import pickle
from tqdm import tqdm
from rdkit.Chem.rdchem import *
from rdkit.Chem.rdmolops import *
from rdkit.Chem.rdmolfiles import *
from sklearn.model_selection import train_test_split
import os
from multiprocessing import Process
# %%
active_keys_file = "active_keys.pkl"
inactive_keys_file = "inactive_keys.pkl"

active_keys = pickle.load(open(active_keys_file, "rb"))
active_keys = [x + "_CHEMBL" for x in active_keys]

inactive_keys = pickle.load(open(inactive_keys_file, "rb"))
inactive_keys = [x + "_ZINC" for x in inactive_keys]

# %%
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

factory = ChemicalFeaturesFactory.get_instance()

def get_feature_dict(mol):
    if mol == None:
        return {}
        
    feature_by_group = {}
    for f in factory.GetFeaturesForMol(mol):
        feature_by_group[f.GetAtomIds()] = f.GetFamily()

    feature_dict = {}
    for key in feature_by_group:
        for atom_idx in key:
            if atom_idx in feature_dict:
                feature_dict[atom_idx].append(feature_by_group[key])
            else:
                feature_dict[atom_idx] = [feature_by_group[key]]

    return feature_dict

# %%
general_dir = "v2020-other-PL"
refined_dir = "refined-set"

general_cpx = os.listdir(general_dir)
refined_cpx = os.listdir(refined_dir)

all_keys = active_keys + inactive_keys

def process_keys(keys):
    for key in tqdm(keys):
        cpx_name = key.split("_")[0]
        load_dir = None
        if cpx_name in general_cpx:
            load_dir = general_dir
        elif cpx_name in refined_cpx:
            load_dir = refined_dir

        # Load ligand
        try_mol2 = False
        try:
            ligands_sdf = SDMolSupplier("%s/%s/%s_ligand.sdf" % (load_dir, cpx_name, cpx_name))
            ligand = ligands_sdf[0]
            if ligand == None:
                try_mol2 = True
                # print("ligand %s" % cpx_name, ligand != None)
        except:
            try_mol2 = True
            print("Error at %s" % cpx_name)

        if try_mol2:
            ligand = MolFromMol2File("%s/%s/%s_ligand.mol2" % (load_dir, cpx_name, cpx_name))
            if ligand == None:
                print("Error ligand %s" % cpx_name, ligand != None)

        ligand_feature = get_feature_dict(ligand)

        # Load receptor
        receptor = MolFromPDBFile("%s/%s/%s_pocket.pdb" % (load_dir, cpx_name, cpx_name))
        # print("receptor %s" % cpx_name, receptor != None)
        receptor_feature = get_feature_dict(receptor)

        pickle.dump(
            [ligand, receptor, ligand_feature, receptor_feature],
            open(
                "../data/%s" % key,
                "wb"
            )
        )

def main():
    list_processes = []

    batch_size = int(len(all_keys) / os.cpu_count()) + 1
    start_idx = 0
    stop_idx = start_idx + batch_size

    for idx in range(os.cpu_count()):
        list_processes.append(Process(target=process_keys, 
                                    args=(all_keys[start_idx:stop_idx],)))

        start_idx = stop_idx
        stop_idx += batch_size
        if stop_idx > len(all_keys):
            stop_idx = len(all_keys)

    for idx in range(len(list_processes)):
        list_processes[idx].start()

    for idx in range(len(list_processes)):
        list_processes[idx].join()

# %%
if __name__ == '__main__':
    main()