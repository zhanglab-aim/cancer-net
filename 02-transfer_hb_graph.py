'''
Keep small portion of gene based on threshold
'''

import numpy as np
import deepdish as dd
from collections import defaultdict

import os
from os import listdir
from os.path import isfile, join
import h5py
import gzip
import pickle
import time


import matplotlib.pyplot as plt
import pandas as pd

import pdb


# ### this part is very time consuming, around 10k seconds###
start_time = time.time()
if os.path.exists('graph/brain_org_network.pickle'):
    with open('graph/brain_org_network.pickle', 'rb') as f:
        edge_dict = pickle.load(f)
    f.close()
else:
    file_brain = 'graph/brain.geneSymbol.gz'
    edge_dict = defaultdict(dict)
    with gzip.open(file_brain, 'rb') as f:
        file_content = f.read()
        for x in file_content.split(b"\n")[:-1]:
            edge_dict[x.split(b'\t')[0].decode('ascii')][x.split(b'\t')[1].decode('ascii')] = float(x.split(b'\t')[2])
            edge_dict[x.split(b'\t')[1].decode('ascii')][x.split(b'\t')[0].decode('ascii')] = float(x.split(b'\t')[2])
        f.close()
    print("--- %s seconds ---" % (time.time() - start_time))
    with open('graph/brain_org_network.pickle', 'wb') as f:
        pickle.dump(edge_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()

