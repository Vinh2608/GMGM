from ic50_statistic import get_kds
import pickle

def main():
    id_kd = get_kds('INDEX_general_PL.2020')
    active_keys = [i for i in id_kd if id_kd[i]<=2.5]
    inactive_keys = [i for i in id_kd if id_kd[i]>2.5]
    pickle.dump(active_keys, open('active_keys.pkl', 'wb'))
    pickle.dump(inactive_keys, open('inactive_keys.pkl', 'wb'))

if __name__=="__main__":
    main()