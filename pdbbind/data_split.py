from ic50_statistic import get_kds
from sklearn.model_selection import train_test_split
import pickle
import sys

active_keys_file = "active_keys.pkl"
inactive_keys_file = "inactive_keys.pkl"

def main():
    # INDEX_general_PL.2020
    id_kd = get_kds(sys.argv[1])
    active_keys = [i for i in id_kd if id_kd[i]<=2.5]
    inactive_keys = [i for i in id_kd if id_kd[i]>2.5]

    pickle.dump(active_keys, open(active_keys_file, 'wb'))
    pickle.dump(inactive_keys, open(inactive_keys_file, 'wb'))

    active_keys = [x + "_CHEMBL" for x in active_keys]
    inactive_keys = [x + "_ZINC" for x in inactive_keys]

    train_keys, test_keys = train_test_split(active_keys+inactive_keys, test_size=0.2, random_state=42)

    with open("../keys/train_pdbbind.pkl", 'wb') as f:
        pickle.dump(train_keys, f)
        
    with open("../keys/test_pdbbind.pkl", 'wb') as f:
        pickle.dump(test_keys, f)

if __name__=="__main__":
    main()