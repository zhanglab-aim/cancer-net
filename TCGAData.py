import os.path as osp
import time

import pickle
import gzip

from collections import defaultdict

import torch
from torch_geometric.data import InMemoryDataset
from build_graph_ref import read_data


class TCGADataset(InMemoryDataset):
    def __init__(
        self,
        root,
        files,
        label_mapping,
        transform=None,
        pre_transform=None,
        name="tcga",
        suffix="",
        gene_graph="gene_graph.gz",
    ):
        self.name = name
        self.suffix = suffix
        self.gene_graph = gene_graph

        if isinstance(files, (list, tuple)):
            self.files = files
        else:
            with open(files, "rt") as f:
                self.files = [_.strip() for _ in f.readlines()]
                self.files = [_ for _ in self.files if len(_) > 0]

        if isinstance(label_mapping, dict):
            self.label_mapping = label_mapping
        else:
            self.label_mapping = {_: i for i, _ in enumerate(label_mapping)}

        super(TCGADataset, self).__init__(root, transform, pre_transform)

        # the base-class constructor generates the processed file if it is missing
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def processed_dir(self) -> str:
        processed_name = "processed"
        if len(self.suffix) > 0:
            processed_name += "_" + self.suffix
        return osp.join(self.root, processed_name)

    def process(self):
        # this only gets called if a saved verison of the processed dataset is not found

        # load gene graph (e.g., from HumanBase)
        graph_file = osp.join(self.root, self.gene_graph)
        graph_noext, _ = osp.splitext(graph_file)
        graph_pickle = graph_noext + ".pkl"

        start_time = time.time()
        if osp.exists(graph_pickle):
            # load pre-parsed version
            with open(graph_pickle, "rb") as f:
                edge_dict = pickle.load(f)
        else:
            # parse the tab-separated file
            edge_dict = defaultdict(dict)
            with gzip.open(graph_file, "rt") as f:
                for line in f:
                    elems = line.strip().split("\t")
                    if len(elems) == 0:
                        continue

                    assert len(elems) == 3

                    # symmetrize, since the initial graph contains edges in only one dir
                    edge_dict[elems[0]][elems[1]] = float(elems[2])
                    edge_dict[elems[1]][elems[0]] = float(elems[2])

            # save pickle for faster loading next time
            with open(graph_pickle, "wb") as f:
                pickle.dump(edge_dict, f, pickle.HIGHEST_PROTOCOL)

        print(f"loading gene graph took {time.time() - start_time:.2f} seconds.")

        # load the data
        full_file_names = [osp.join(self.raw_dir, _) for _ in self.raw_file_names]
        self.data, self.slices = read_data(
            full_file_names, edge_dict, self.label_mapping
        )

        # pre-filter and/or pre-transform, if necessary
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        # save the processed dataset for later use
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return "TCGADataset(name={}, len={}, suffix={})".format(
            self.name, len(self), self.suffix
        )
