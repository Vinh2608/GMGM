from ic50_statistic import get_kds
from sklearn.model_selection import train_test_split
import pickle
import sys

keys_file = "keys.pkl"
#active_keys_file = "active_keys.pkl"
#inactive_keys_file = "inactive_keys.pkl"

def main():
    # INDEX_general_PL.2020
    id_kd = get_kds(sys.argv[1])
    #active_keys = [i for i in id_kd if id_kd[i]<=2.5]
    keys = [i for i in id_kd]

    pickle.dump(keys, open(keys_file, 'wb'))
    #pickle.dump(active_keys, open(active_keys_file, 'wb'))
    #pickle.dump(inactive_keys, open(inactive_keys_file, 'wb'))

    keys = [x + "_" + str(id_kd[x]) for x in keys]

    train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=42)

    with open("../keys/train_pdbbind.pkl", 'wb') as f:
        pickle.dump(train_keys, f)
        
    with open("../keys/test_pdbbind.pkl", 'wb') as f:
        pickle.dump(test_keys, f)


if __name__=="__main__":
    main()