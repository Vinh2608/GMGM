# %%
import os
import pickle
from tqdm import tqdm
from rdkit.Chem.rdchem import *
from rdkit.Chem.rdmolops import *
from rdkit.Chem.rdmolfiles import *
from sklearn.model_selection import train_test_split

# %%
active_keys_file = "active_keys.pkl"
inactive_keys_file = "inactive_keys.pkl"

active_keys = pickle.load(open(active_keys_file, "rb"))
active_keys = [x + "_CHEMBL" for x in active_keys]

inactive_keys = pickle.load(open(inactive_keys_file, "rb"))
inactive_keys = [x + "_ZINC" for x in inactive_keys]

# %%
general_dir = "v2020-other-PL"
refined_dir = "refined-set"

general_cpx = os.listdir(general_dir)
refined_cpx = os.listdir(refined_dir)

all_keys = active_keys + inactive_keys
for key in tqdm(all_keys):
    cpx_name = key.split("_")[0]
    load_dir = None
    if cpx_name in general_cpx:
        load_dir = general_dir
    elif cpx_name in refined_cpx:
        load_dir = refined_dir

    # Load ligand
    ligands_sdf = SDMolSupplier("%s/%s/%s_ligand.sdf" % (load_dir, cpx_name, cpx_name))
    ligand = ligands_sdf[0]
    # print("ligand %s" % cpx_name, ligand != None)
    
    # Load receptor
    receptor = MolFromPDBFile("%s/%s/%s_protein.pdb" % (load_dir, cpx_name, cpx_name))
    # print("receptor %s" % cpx_name, receptor != None)

    pickle.dump(
        [ligand, receptor],
        open(
            "../data/%s" % key,
            "wb"
        )
    )
