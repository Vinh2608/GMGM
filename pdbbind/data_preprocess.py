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
        
        # Load receptor
        receptor = MolFromPDBFile("%s/%s/%s_pocket.pdb" % (load_dir, cpx_name, cpx_name))
        # print("receptor %s" % cpx_name, receptor != None)

        pickle.dump(
            [ligand, receptor],
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