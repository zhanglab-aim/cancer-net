"""
TCGAData.py calls the functions in this script. The threshold is hard coding here
"""

from logging import warning
import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing

# import billiard as multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from random import shuffle
from scipy.stats import kurtosis, skew
import deepdish as dd
from functools import partial
from itertools import repeat

from os import listdir
from os.path import isfile, join
import pickle
import h5py
import gzip
import time
import os
from collections import defaultdict

import mygene

# ---------- Load Preprocessed ---------#
# ---- RUN 02-transfer_hb_graph.py which saves the filtered 'graph/brain_th{}.h5' -----#
# load ref gene feature
import h5py

dataroot = os.path.join(os.path.dirname(__file__), "data")
f = h5py.File(dataroot + "/raw/ref_gen.h5")
embed = f["embed"]
ref_fea0 = embed["embed_block_0"][()]
ref_fea1 = embed["embed_block_1"][()]
gene_name = np.char.decode(f["meta"]["gene_symbol"][()])
f.close()
ref_fea = np.concatenate([ref_fea0, ref_fea1], axis=1)

# ------ Read column names from file
# path = '/data/raw'
# path = '../../data/frank/embedded/raw'
path = os.path.join(dataroot, "raw")
samples = [f for f in listdir(path) if isfile(join(path, f))]


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

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


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.


def read_data(samples, edge_dict, label_mapping):
    batch = []
    num_node_list = []
    y_list = []
    edge_att_list, edge_index_list, att_list = [], [], []
    res = []

    start = time.time()
    for s in samples:
        try:
            a = read_single_data(s, edge_dict, label_mapping)
            if a is None:
                print("invalid sample: %s" % s)
                continue
            else:
                res.append(a)
        except Exception as e:
            print("error while processing %s" % s)
            raise e

    stop = time.time()

    print(f"Loading dataset took {stop - start:.2f} seconds.")

    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        if j == 0:
            edge_index_list.append(res[j][1])
        else:
            edge_index_list.append(res[j][1] + sum(num_node_list))
        num_node_list.append(res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j] * res[j][4])

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(
        edge_att_arr.reshape(len(edge_att_arr), 1)
    ).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    data = Data(
        x=att_torch,
        edge_index=edge_index_torch,
        y=y_torch,
        edge_attr=edge_att_torch.squeeze(),
    )

    data, slices = split(data, batch_torch)

    return data, slices


def read_single_data(sample, edge_dict, label_mapping, only_mutated=True):
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
        # find the gene without any mutation in fea
        EPS = 1e-8

        diff = np.sum(np.abs(ref_fea - fea), axis=1)
        mu_id = np.where(diff > EPS)[0]
        if len(mu_id) == 0:
            return None

        # get node features
        att = fea[mu_id, :]  # alternatively, we can do fea[mu_id,:] - ref_fea[mu_id,:]

        # construct graph structure

        # ----first we  select the mutated gene name from TCGA's gene names ---#
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
        mask = ["symbol" in _ for _ in mg_results]
        att = fea[mask, :]

    # --- third match assign the HB links to TCGA mutated genes ---#
    cmat = np.zeros((len(mu_gene_name), len(mu_gene_name)))
    for i, gname1 in enumerate(mu_gene_name):
        for j, gname2 in enumerate(mu_gene_name):
            try:
                cmat[i, j] = edge_dict[gname1][gname2]
            except KeyError:
                continue

    # ---- fourth Remove small edges -----#
    # edge_th = np.percentile(cmat.reshape(-1), 90)
    edge_th = 0.1
    cmat[cmat < edge_th] = 0
    # import pdb
    # pdb.set_trace()
    num_nodes = cmat.shape[0]
    # ------- fifth build a graph --------#
    cmat = cmat + np.identity(cmat.shape[0])
    G = from_numpy_matrix(cmat)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = cmat[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(
        torch.from_numpy(edge_index), torch.from_numpy(edge_att)
    )
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)

    # get graph labels
    label_str = data["label"]["sample_meta"]["tumor"][()]
    label = label_mapping[label_str]

    data.close()
    return edge_att.data.numpy(), edge_index.data.numpy(), att, label, num_nodes


if __name__ == "__main__":
    start_time = time.time()
    dataroot = os.path.dirname(__file__)
    if os.path.exists(dataroot + "/graph/brain_org_network.pickle"):
        with open(dataroot + "/graph/brain_org_network.pickle", "rb") as f:
            edge_dict = pickle.load(f)
        f.close()
    else:
        start_time = time.time()
        file_brain = dataroot + "/graph/brain.geneSymbol.gz"
        edge_dict = defaultdict(dict)
        with gzip.open(file_brain, "rb") as f:
            file_content = f.read()
            for x in file_content.split(b"\n")[:-1]:
                edge_dict[x.split(b"\t")[0].decode("ascii")][
                    x.split(b"\t")[1].decode("ascii")
                ] = float(x.split(b"\t")[2])
                edge_dict[x.split(b"\t")[1].decode("ascii")][
                    x.split(b"\t")[0].decode("ascii")
                ] = float(x.split(b"\t")[2])
            f.close()
        with open("graph/brain_org_network.pickle", "wb") as f:
            pickle.dump(edge_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    print(
        "--- load human base brain network %s seconds ---" % (time.time() - start_time)
    )
    # read_data(samples)
    start_time = time.time()
    for s in samples:
        try:
            read_single_data(s, edge_dict)
        except nx.exception.NetworkXError:
            print(s)
    print("--- read file %s seconds ---" % (time.time() - start_time))

