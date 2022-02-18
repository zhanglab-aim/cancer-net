from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from os import listdir
import os.path as osp
from build_graph_ref import read_data

import torch
import numpy as np
import pandas as pd
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import deepdish as dd
from os import listdir
from os.path import isfile, join

import time
import pickle
import h5py
import gzip
import time
import os
from collections import defaultdict

dataroot = os.path.join(os.path.dirname(__file__))

class TCGADataset(InMemoryDataset):
    # TT: self.name was missing, making __repr__ invalid
    def __init__(self, root, transform=None, pre_transform=None, name="tcga"):
        super(TCGADataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.name = name
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        # we load gene names from Human Base #
        start_time = time.time()
        if os.path.exists(dataroot+'/graph/brain_org_network.pickle'):
            with open(dataroot+'/graph/brain_org_network.pickle', 'rb') as f:
                edge_dict = pickle.load(f)
            f.close()
        else:
            start_time = time.time()
            file_brain = dataroot + './graph/brain.geneSymbol.gz'
            edge_dict = defaultdict(dict)
            with gzip.open(file_brain, 'rb') as f:
                file_content = f.read()
                for x in file_content.split(b"\n")[:-1]:
                    edge_dict[x.split(b'\t')[0].decode('ascii')][x.split(b'\t')[1].decode('ascii')] = float(
                        x.split(b'\t')[2])
                    edge_dict[x.split(b'\t')[1].decode('ascii')][x.split(b'\t')[0].decode('ascii')] = float(
                        x.split(b'\t')[2])
                f.close()
            with open(dataroot + './graph/brain_org_network.pickle', 'wb') as f:
                pickle.dump(edge_dict, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        print("--- load human base brain network %s seconds ---" % (time.time() - start_time))

        self.samples = [f for f in listdir(join(self.root, 'raw')) if isfile(join(self.root, 'raw', f)) and f.startswith('TCGA') ]
        self.data, self.slices = read_data(self.samples,edge_dict)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
