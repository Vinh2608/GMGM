import os
import re
from pdbbind.data_split import get_kds

file_name = get_kds('INDEX_general_PL.2020')

for file_name1 in file_name.keys():
    for file_name2 in os.listdir('data'):
        indexFind = file_name2.find('_')
        if (file_name2[0:indexFind] == file_name1):
            os.rename('data/' + file_name2, 'data/' + file_name1 + "_" + str(file_name[file_name1]))