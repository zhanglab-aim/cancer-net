"""Some helper code for reading TCGA datasets."""

import os
import time
import tqdm

from typing import Tuple

from logging import warning

import h5py

import numpy as np
from scipy import sparse
import torch
from torch_geometric.data import Data
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops

import mygene

ref_fea = None


def split(data: Data, batch: torch.Tensor) -> Tuple[Data, dict]:
    """Split data into slices.

    This changes `data` in-place! XXX why?
    
    :param data: data in `torch_geometric` `Data` format
    :param batch: sequence of sample indices for each entry in data
    :return: tuple `(data, slices)`, the former returns the input `data` structure after
        ensuring that `edge_index` starts at 0 for each sample; and the latter is a
        `dict` containing the following if the corresponding fields exist in `data`,
        with the exception of `pos`, which is always there
            edge_index: starting positions for each sample in edge list
            edge_attr:  starting positions for each sample in edge list
            x:          starting positions for each sample in node list
            y:          starting positions for each sample in edge list, if `y` has the
                        same size as `batch`, otherwise the numbers 0, ...,
                        `batch[-1] + 1`; XXX why?
            pos:        starting positions for each sample in node list
    """
    # make tensor of starting positions for each sample in node list
    # `np.bincount` automatically sorts these in increasing order of sample indices
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    # make tensor of starting positions for each sample in edge list
    # (this uses only row information because we know edges are only within each sample)
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # recover indices starting at zero for each sample
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    # create the slices dict
    slices = {"edge_index": edge_slice}
    if data.x is not None:
        slices["x"] = node_slice
    if data.edge_attr is not None:
        slices["edge_attr"] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices["y"] = node_slice
        else:
            slices["y"] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices["pos"] = node_slice

    return data, slices


def read_data(samples: list, edge_dict: dict, label_mapping: dict) -> Tuple[Data, dict]:
    """Read a set of TCGA samples.
    
    :param samples: list of data file names
    :param edge_dict: graph of gene interactions
    :param label_mapping: mapping from `bytes`/`str` label encodings to numeric values
    :return: tuple `(data, slices)`, the former as a `torch_geometric` `Data` object,
        the later as a `dict`
    """
    res = []

    # read the sample data from file
    start = time.time()
    for s in tqdm.tqdm(samples, desc="read samples"):
        try:
            a = read_single_data(s, edge_dict, label_mapping)
            # skip invalid samples
            if a is None:
                print(f"invalid sample: {s}")
                continue
            else:
                res.append(a)
        except Exception as e:
            print(f"error while processing {s}")
            raise e

    stop = time.time()
    print(f"Loading dataset took {stop - start:.2f} seconds.")

    # concatenate all the sample data into one big tensor
    batch = []  # keeps track of the sample index
    num_node_list = []  # keeps track of the number of nodes per sample
    y_list = []  # keeps track of labels

    # these keep track of the graph: edge attributes, edge vertices, node attributes
    edge_att_list, edge_index_list, att_list = [], [], []

    # this first makes lists of lists (or arrays)
    for j in range(len(res)):
        # update vertex indices to refer to concatenated lists
        edge_index_list.append(res[j][1] + sum(num_node_list))

        # the rest can just be copied over
        edge_att_list.append(res[j][0])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        num_node_list.append(res[j][4])

        batch.append([j] * res[j][4])

    # make the concatenated arrays
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    edge_att_arr = np.concatenate(edge_att_list)
    att_arr = np.concatenate(att_list, axis=0)
    y_arr = np.stack(y_list)

    # convert to PyTorch
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    edge_att_torch = torch.from_numpy(
        edge_att_arr.reshape(len(edge_att_arr), 1)
    ).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()

    # make a torch_geometric Data object
    data = Data(
        edge_index=edge_index_torch,
        edge_attr=edge_att_torch.squeeze(),
        x=att_torch,
        y=y_torch,
    )

    # generate slices
    data, slices = split(data, batch_torch)

    return data, slices


def read_single_data(
    sample,
    edge_dict,
    label_mapping,
    edge_tol: bool = 0.1,
    only_mutated: bool = True,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Read a single TCGA sample file.
    
    :param sample: name of HDF file
    :param edge_dict: graph of gene interactions
    :param label_mapping: mapping from `bytes`/`str` label encodings to numeric values
    :param edge_tol: edges with weights smaller than this threshold are pruned
    :return: tuple:
        edge_att:   tensor of edge attributes, shape `(n_edge,)`
        edge_index: tensor of vertex indices for each edge, shape `(2, n_edge)`
        att:        array of node attributes
        label:      numeric sample label
        num_nodes:  number of nodes in the graph
    """
    # open and read the HDF file
    try:
        data = h5py.File(sample, "r")
    except OSError:
        warning(f"Cannot load sample file: {sample}.")

    if "data" not in data:
        return None

    fea0 = data["data"]["promoter"][()]  # 64-column, noncoding features
    fea1 = data["data"]["protein"][()]  # 64-column, coding features
    fea = np.concatenate([fea0, fea1], axis=1)

    if not only_mutated:
        # XXX this is hacky and is currently not used
        if ref_fea is None:
            # load the reference feature encodings
            dataroot = os.path.join(os.path.dirname(__file__), "data")
            f = h5py.File(os.path.join(dataroot, "raw", "ref_gen.h5"))
            embed = f["embed"]
            ref_fea0 = embed["embed_block_0"][()]
            ref_fea1 = embed["embed_block_1"][()]
            gene_name = np.char.decode(f["meta"]["gene_symbol"][()])
            f.close()
            ref_fea = np.concatenate([ref_fea0, ref_fea1], axis=1)

        # find mutated genes by comparing to reference
        diff = np.sum(np.abs(ref_fea - fea), axis=1)
        mu_id = np.where(diff > tol)[0]
        if len(mu_id) == 0:
            return None

        # get node features and gene names
        att = fea[mu_id, :]  # alternatively, we can do fea[mu_id,:] - ref_fea[mu_id,:]
        mu_gene_name = gene_name[mu_id]
    else:
        # get the list of mutated genes in Ensembl format
        # need to convert bytes to str and remove the version number (part after ".")
        ensembl_gene_names = [
            _.decode("utf-8").split(".")[0]
            for _ in data["meta"]["mutated_gene_list"][()]
        ]

        # now convert to HGNC symbols
        mg = mygene.MyGeneInfo()
        mg_results = mg.querymany(ensembl_gene_names, fields="symbol", verbose=False)

        hgnc_names = [_["symbol"] for _ in mg_results if "symbol" in _]
        mu_gene_name = hgnc_names

        # some gene names are not found -- not entirely sure why; errors in the data?
        # skip them
        mask = ["symbol" in _ for _ in mg_results]
        att = fea[mask, :]

    # construct graph structure, using `edge_dict` to generated edges
    cmat = np.zeros((len(mu_gene_name), len(mu_gene_name)))
    for i, gname1 in enumerate(mu_gene_name):
        for j, gname2 in enumerate(mu_gene_name):
            try:
                cmat[i, j] = edge_dict[gname1][gname2]
            except KeyError:
                continue

    # remove edges with very small weights
    # edge_th = np.percentile(cmat.reshape(-1), 90)
    cmat[cmat < edge_tol] = 0
    num_nodes = cmat.shape[0]

    # convert graph to encoding useful for torch_geometric -- lists of edges and attrs
    adj = sparse.coo_matrix(cmat)

    edge_index = np.stack([adj.row, adj.col])
    edge_att = adj.data

    # ensure there are no self loops
    edge_index, edge_att = remove_self_loops(
        torch.from_numpy(edge_index), torch.from_numpy(edge_att)
    )
    edge_index = edge_index.long()

    # sort indices and add weights for duplicated edges (we don't have any here, though)
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)

    # assign a numeric label
    label_bytes = data["label"]["sample_meta"]["tumor"][()]
    try:
        label = label_mapping[label_bytes]
    except KeyError:
        label = label_mapping[label_bytes.decode("utf-8")]

    # close the HDF file and return
    data.close()
    return edge_att.data.numpy(), edge_index.data.numpy(), att, label, num_nodes
