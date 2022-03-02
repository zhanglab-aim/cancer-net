import os
import os.path as osp
import time
import tqdm

from typing import Union, Optional, Callable

import pickle
import gzip

from collections import defaultdict

import torch
from torch_geometric.data import InMemoryDataset
from build_graph_ref import read_data


class TCGADataset(InMemoryDataset):
    """A dataset of samples from The Cancer Genome Atlas (TCGA).

    This reads TCGA data files in HDF format from a given `root` folder. The file names
    are either explicitly given in the constructor, or are read from a text file (see
    `files` below).
    
    Each TCGA file is expected to contain an encoded version of the mutated genes from
    each sample, together with a string label. This label is converted to an integer
    using the `label_mapping` (see below). The data is converted to a graph in which
    each mutated gene is a node, with the encoded data from the TCGA file used as node
    attributes. The edge identities and weights are copied from the `gene_graph` (see
    below). The data is `pre_transform`ed after loading, and then saved in a subfolder
    of the root folder whose name starts with "processed" and is potentially followed by
    "_" and the `suffix`, if there is one. The filename used for the processed data is
    "data.pt". The `pre_transform` and `pre_filter` that were used are also stored in
    this folder, with names `"pre_transform.pt"` and `"pre_filter.pt"`, respectively.
    (`pre_filter` is actually not currently used here, but `torch_geometric`'s `Dataset`
    generates the `.pt` file anyway.)

    The `gene_graph` is expected to be a gzip-compressed tab-separated file with rows of
    the form `(gene_symbol1, gene_symbol2, weight)`, encoding a graph. This is read and
    symmetrized (i.e., edges are added for both `(gene_symbol1, gene_symbol2)` and
    `(gene_symbol2, gene_symbol1`)), and the result is stored in a pickle in the same
    folder as the gzip file, for faster access in future runs.
    
    :param root: root folder for the samples; also used to store processed data
    :param label_mapping: dictionary from sample labels in `str` or `bytes` format to
        numeric labels; this can also be a list, in which case each element is mapped to
        its index in the list
    :param files: either a list of data file names (in HDF format), or the name of a
        single text file containing the names of the data files; by default, the list of
        files names is loaded from a file called `samples.txt` in the root folder (but
        see the `samples_file` option)
    :param transform: transformation applied at each access
    :param pre_transform: transformation applied before saving to disk
    :param name: name for the dataset; default: "tcga_" followed by the root's
        `basename` and the `suffix`, if there is one, preceded by "_"
    :param suffix: suffix used for generating the name of the processed-file folder, and
        also for setting the default `name`
    :param gene_graph: filename for the gene graph in the `graph_dir` folder
    :param graph_dir: subfolder of root where to look for the gene graph
    :param samples_file: file in the root folder where the sample names are read from
    """

    def __init__(
        self,
        root: str,
        label_mapping: Union[list, dict],
        files: Union[str, list, None] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        name: Optional[str] = None,
        suffix: str = "",
        gene_graph: str = "gene_graph.gz",
        graph_dir: str = "graph",
        samples_file: str = "samples.txt",
    ):
        self.suffix = suffix
        self.gene_graph = gene_graph
        self.graph_dir = graph_dir

        # handle defaults
        if name is not None:
            self.name = name
        else:
            self.name = "tcga_" + osp.basename(root)
            if len(suffix) > 0:
                self.name = self.name + "_" + suffix

        if files is None:
            files = osp.join(root, samples_file)

        if isinstance(files, (list, tuple)):
            self.files = files
        else:
            # read list of files from file
            with open(files, "rt") as f:
                self.files = [_.strip() for _ in f.readlines()]
                self.files = [_ for _ in self.files if len(_) > 0]

        # handle the two formats for the label mapping
        if isinstance(label_mapping, dict):
            self.label_mapping = label_mapping
        else:
            self.label_mapping = {_: i for i, _ in enumerate(label_mapping)}

        # the base-class constructor will generate the processed file if it is missing
        super(TCGADataset, self).__init__(root, transform, pre_transform)

        # load the processed dataset
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def processed_dir(self) -> str:
        # the name of the processed folder can be changed by using the `suffix`
        processed_name = "processed"
        if len(self.suffix) > 0:
            processed_name += "_" + self.suffix
        return osp.join(self.root, processed_name)

    def process(self):
        # this only gets called if a saved verison of the processed dataset is not found

        start_time = time.time()

        # load gene graph (e.g., from HumanBase)
        graph_file = osp.join(self.root, self.graph_dir, self.gene_graph)
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
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                f.seek(0, os.SEEK_SET)

                pbar = tqdm.tqdm(
                    total=file_size,
                    unit_scale=True,
                    unit_divisor=1024,
                    mininterval=1.0,
                    desc="gene graph",
                )
                for line in f:
                    pbar.update(len(line))
                    elems = line.strip().split("\t")
                    if len(elems) == 0:
                        continue

                    assert len(elems) == 3

                    # symmetrize, since the initial graph contains edges in only one dir
                    edge_dict[elems[0]][elems[1]] = float(elems[2])
                    edge_dict[elems[1]][elems[0]] = float(elems[2])

                pbar.close()

            # save pickle for faster loading next time
            t0 = time.time()
            print("Caching the graph as a pickle...", end=None)
            with open(graph_pickle, "wb") as f:
                pickle.dump(edge_dict, f, pickle.HIGHEST_PROTOCOL)
            print(f" done (took {time.time() - t0:.2f} seconds).")

        print(f"loading gene graph took {time.time() - start_time:.2f} seconds.")

        # load the data
        full_file_names = [osp.join(self.raw_dir, _) for _ in self.raw_file_names]
        data_list = read_data(full_file_names, edge_dict, self.label_mapping)

        # pre-filter and/or pre-transform, if necessary
        if self.pre_filter is not None:
            t0 = time.time()
            data_list = [data for data in data_list if self.pre_filter(data)]
            print(f"Pre-filtering took {time.time() - t0:.2f} seconds.")

        if self.pre_transform is not None:
            t0 = time.time()
            data_list = [self.pre_transform(data) for data in data_list]
            print(f"Pre-transforming took {time.time() - t0:.2f} seconds.")

        self.data, self.slices = self.collate(data_list)

        # save the processed dataset for later use
        t0 = time.time()
        print("Caching processed dataset...", end=None)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print(f" done (took {time.time() - t0:.2f} seconds).")

        print(f"Full processing pipeline took {time.time() - start_time:.2f} seconds.")

    def __repr__(self):
        return f'TCGADataset(name={self.name}, len={len(self)}, suffix="{self.suffix}")'
