import re
import numpy as np
from matplotlib import pyplot as plt


def get_kds(file_path):
    lines = open(file_path).readlines()

    # remove intro lines
    lines = lines[6:]   
    # split data fields
    lines = [[j for j in i.split(' ') if j!=''] for i in lines]     
    # get Kd/Ki/IC50 field
    kds = [i[3] for i in lines]     
    # remove prefix, get value only
    kds = [re.search('[\d\.]+[a-zA-Z]+', i).group() for i in kds]
    # get the unit mM, uM, nM, pM, fM
    units = [re.search('[^\d\.]+', i).group() for i in kds]
    # dict to convert all to uM
    unit_convert = {'mM': 1e3, 'uM': 1, 'nM': 1e-3, 'pM': 1e-6, 'fM': 1e-9}
    # get float values 
    values = [float(re.search('[\d\.]+', i).group()) for i in kds] 
    # convert to 1 unit uM
    values = [v*unit_convert[units[i]] for i, v in enumerate(values)]

    ids = [i[0] for i in lines]

    return dict([(ids[i], values[i]) for i in range(len(ids))])

if __name__ == "__main__":

    values = get_kds('INDEX_general_PL.2020').values()

    plt.rcParams["figure.figsize"] = (20,10)

    ###
    tmp = [i for i in values if i <= 30]
    # tmp = values
    val_plot, base = np.histogram(tmp, bins=300)
    cumulative = np.cumsum(val_plot)/len(values)*100

    base, cumulative = list(base), list(cumulative)

    plt.plot([0]+base[:-1], [0]+cumulative, c='blue')
    plt.xticks(np.arange(0,30.1, 1))
    plt.yticks(np.arange(0,100.1, 10))
    plt.xlabel("Ki/Kd/IC50 (uM)")
    plt.ylabel("%")
    plt.savefig('0_30.jpg')
    plt.show()


    ###
    tmp = [i for i in values if i <= 1]
    val_plot, base = np.histogram(tmp, bins=1000)
    cumulative = np.cumsum(val_plot)/len(values)*100

    base, cumulative = list(base), list(cumulative)

    plt.plot([0]+base[:-1], [0]+cumulative, c='blue')
    plt.xticks(np.arange(0,1.0001, 0.1))
    plt.yticks(np.arange(0,100.1, 10))
    plt.xlabel("Ki/Kd/IC50 (uM)")
    plt.ylabel("%")
    plt.savefig('0_1.jpg')
    plt.show()

    ###
    tmp = [i for i in values if i <= 30]
    w = [1/len(values)]*len(tmp)
    plt.hist(tmp, bins=30, weights=w)
    plt.xticks(np.arange(0,30.1, 1))
    plt.xlabel("Ki/Kd/IC50 (uM)")
    plt.savefig('0_30_hist.jpg')
    plt.show()

    ###
    tmp = [i for i in values if i <= 1]
    w = [1/len(values)]*len(tmp)
    plt.hist(tmp, bins=10, weights=w)
    plt.xticks(np.arange(0,1.0001, 0.1))
    plt.xlabel("Ki/Kd/IC50 (uM)")
    plt.savefig('0_1_hist.jpg')
    plt.show()