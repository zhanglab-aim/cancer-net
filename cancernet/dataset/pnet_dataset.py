import os
import time

import copy
import gzip
import logging
import pickle
import tqdm

import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Optional, Callable

# for graph building
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

# for Reactome pathways
import networkx as nx
import itertools
import re


cached_data = {}  # all data read will be referenced here


### Pytorch geometric Dataset ###

class PnetDataSet(InMemoryDataset):
    def __init__(
        self,
        name="prostate_graph_humanbase",
        edge_tol=0.5,
        root: Optional[str] = "./data/prostate/",
        files: Optional[dict] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.25,
        valid_seed: int = 0,
        test_seed: int = 7357,
    ):
        # the base-class constructor will generate the processed file if it is missing
        # self.all_data, self.response, self.edge_dict, self.edge_tol = all_data, response, edge_dict, edge_tol
        self.edge_tol = edge_tol
        self.name = name
        self._files = files or {}
        super(PnetDataSet, self).__init__(root, transform, pre_transform)

        # load the processed dataset
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(os.path.join(self.root, "node_index.pkl"), "rb") as f:
            self.node_index = pickle.load(f)
        # call pyg
        self.num_samples = len(self.data.y)
        self.num_test_samples = int(test_ratio * self.num_samples)
        self.num_valid_samples = int(valid_ratio * self.num_samples)
        self.num_train_samples = (
            self.num_samples - self.num_test_samples - self.num_valid_samples
        )
        self.split_index_by_rng(test_seed=test_seed, valid_seed=valid_seed)

    def process(self):
        t0 = time.time()
        all_data, response, edge_dict = data_reader(filename_dict=self.raw_file_names)
        print(f"read raw data took {time.time()-t0:.2f} seconds")
        # response = pnet_utils.cached_data['response'].loc[all_data.index]
        self.response, self.edge_dict = response, edge_dict
        edge_tol = self.edge_tol
        # match inputs
        hb_genes = sorted([x for x in all_data.columns.levels[0] if x in edge_dict])
        self.all_data = all_data[hb_genes]
        node_index = {g: i for i, g in enumerate(hb_genes)}
        self.node_index = node_index
        with open(os.path.join(self.root, "node_index.pkl"), "wb") as f:
            pickle.dump(node_index, f)
        # convert and filter edges
        edge_index = []
        edge_att = []
        for a in tqdm.tqdm(edge_dict, desc="filter by edge_tol"):
            if not a in node_index:
                continue
            for b in edge_dict[a]:
                if not b in node_index:
                    continue
                w = edge_dict[a][b]
                if w > edge_tol:
                    edge_index.append((node_index[a], node_index[b]))
                    edge_att.append(w)

        # ensure there are no self loops
        edge_index = np.array(edge_index).T
        edge_att = np.array(edge_att)
        edge_index, edge_att = remove_self_loops(
            torch.from_numpy(edge_index), torch.from_numpy(edge_att)
        )
        edge_index = edge_index.long()

        # sort indices and add weights for duplicated edges (we don't have any here, though)
        num_nodes = len(hb_genes)
        edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
        edge_att = edge_att.type(torch.float)

        # build each data
        data_list = []
        for ind in range(self.all_data.shape[0]):
            dat = Data(
                edge_index=edge_index,
                edge_attr=edge_att.type(torch.float),
                x=torch.Tensor(self.all_data.iloc[ind].unstack().to_numpy()),
                y=int(
                    self.response.iloc[ind][0]
                ),  # this place is really weird - if I don't pass int(),
                # it will somehow mess up the slices['y']
                subject_id=str(self.all_data.index[ind]),
            )
            data_list.append(dat)

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
        torch.save((self.data, self.slices), self.processed_paths[0])

    def split_index_by_rng(self, test_seed, valid_seed):
        # train/valid/test random generators
        rng_test = np.random.default_rng(test_seed)
        rng_valid = np.random.default_rng(valid_seed)

        # splitting off the test indices
        test_split_perm = rng_test.permutation(self.num_samples)
        self.test_idx = list(test_split_perm[: self.num_test_samples])
        self.trainvalid_indices = test_split_perm[self.num_test_samples :]

        # splitting off the validation from the remainder
        valid_split_perm = rng_valid.permutation(len(self.trainvalid_indices))
        self.valid_idx = list(
            self.trainvalid_indices[valid_split_perm[: self.num_valid_samples]]
        )
        self.train_idx = list(
            self.trainvalid_indices[valid_split_perm[self.num_valid_samples :]]
        )

    def split_index_by_file(self, train_fp, valid_fp, test_fp):
        train_set = pd.read_csv(train_fp, index_col=0)
        valid_set = pd.read_csv(valid_fp, index_col=0)
        test_set = pd.read_csv(test_fp, index_col=0)
        if not hasattr(self, "response"):  # this is hacky
            # self.response = pnet_utils.cached_data['response']
            self.response = pd.DataFrame(
                {"response": self.data.y}, index=self.data.subject_id
            )
        self.train_idx = [
            self.response.index.get_loc(x)
            for x in train_set.id
            if x in self.response.index
        ]
        self.valid_idx = [
            self.response.index.get_loc(x)
            for x in valid_set.id
            if x in self.response.index
        ]
        self.test_idx = [
            self.response.index.get_loc(x)
            for x in test_set.id
            if x in self.response.index
        ]
        # check no redundency
        assert len(self.train_idx) == len(set(self.train_idx))
        assert len(self.valid_idx) == len(set(self.valid_idx))
        assert len(self.test_idx) == len(set(self.test_idx))
        # check no overlap
        assert len(set(self.train_idx).intersection(set(self.valid_idx))) == 0
        assert len(set(self.train_idx).intersection(set(self.test_idx))) == 0
        assert len(set(self.valid_idx).intersection(set(self.test_idx))) == 0

    def __repr__(self):
        return (
            f"PnetDataset("
            f"len={len(self)}, "
            f"graph={self.name}, "
            f"num_edges={self.get(0).edge_index.shape[1]}, "
            f"edge_tol={self.edge_tol:.2f}"
            f")"
        )

    @property
    def raw_file_names(self):
        return {
            # non-tumor-specific data
            "graph_file": os.path.join(
                self.root, self._files.get("graph_file", "prostate_gland.geneSymbol.gz")
            ),
            "selected_genes": os.path.join(
                self.root,
                self._files.get(
                    "selected_genes",
                    "tcga_prostate_expressed_genes_and_cancer_genes.csv",
                ),
            ),
            "use_coding_genes_only": os.path.join(
                self.root,
                self._files.get(
                    "use_coding_genes_only",
                    "protein-coding_gene_with_coordinate_minimal.txt",
                ),
            ),
            # tumor data
            "response": os.path.join(
                self.root, self._files.get("response", "response_paper.csv")
            ),
            "mut_important": os.path.join(
                self.root,
                self._files.get(
                    "mut_important", "P1000_final_analysis_set_cross_important_only.csv"
                ),
            ),
            "cnv_amp": os.path.join(
                self.root, self._files.get("cnv_amp", "P1000_data_CNA_paper.csv")
            ),
            "cnv_del": os.path.join(
                self.root, self._files.get("cnv_del", "P1000_data_CNA_paper.csv")
            ),
        }

    @property
    def processed_file_names(self):
        return f"data-{self.name}-{self.edge_tol:.2f}.pt"

    @property
    def processed_dir(self) -> str:
        return self.root


def data_reader(filename_dict):
    # sanity checks for filename_dict
    assert "response" in filename_dict, "must parse a response file"
    for f in filename_dict.values():
        if not os.path.isfile(f):
            raise FileNotFoundError(f)
    fd = copy.deepcopy(filename_dict)
    # first get non-tumor genomic/config data types out
    edge_dict = graph_reader_and_processor(graph_file=fd.pop("graph_file"))
    selected_genes = fd.pop("selected_genes")
    if selected_genes is not None:
        selected_genes = pd.read_csv(selected_genes)["genes"]
    use_coding_genes_only = fd.pop("use_coding_genes_only")
    # read the remaining tumor data
    labels = get_response(fd.pop("response"))
    x_list = []
    y_list = []
    rows_list = []
    cols_list = []
    data_type_list = []
    for data_type, filename in fd.items():
        x, y, info, genes = load_data(filename=filename, selected_genes=selected_genes)
        x = processor(x, data_type)
        x_list.append(x), y_list.append(y), rows_list.append(info), cols_list.append(
            genes
        )
        data_type_list.append(data_type)
    res = combine(
        x_list,
        y_list,
        rows_list,
        cols_list,
        data_type_list,
        combine_type="union",
        use_coding_genes_only=use_coding_genes_only,
    )
    all_data = res[0]
    response = labels.loc[all_data.index]
    return all_data, response, edge_dict


def graph_reader_and_processor(graph_file):
    # load gene graph (e.g., from HumanBase)
    # graph_file = os.path.join(self.root, self.graph_dir, self.gene_graph)
    graph_noext, _ = os.path.splitext(graph_file)
    graph_pickle = graph_noext + ".pkl"

    start_time = time.time()
    if os.path.exists(graph_pickle):
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
    return edge_dict


def processor(x, data_type):
    if data_type == "mut_important":
        x[x > 1.0] = 1.0
    elif data_type == "cnv_amp":
        x[x <= 0.0] = 0.0
        x[x == 1.0] = 0.0
        x[x == 2.0] = 1.0
    elif data_type == "cnv_del":
        x[x >= 0.0] = 0.0
        x[x == -1.0] = 0.0
        x[x == -2.0] = 1.0
    else:
        raise TypeError("unknown data type '%s' % data_type")
    return x


def get_response(response_filename):
    logging.info("loading response from %s" % response_filename)
    labels = pd.read_csv(response_filename)
    labels = labels.set_index("id")
    if "response" in cached_data:
        logging.warning(
            "response in cached_data is being overwritten by '%s'" % response_filename
        )
    else:
        logging.warning(
            "response in cached_data is being set by '%s'" % response_filename
        )

    cached_data["response"] = labels
    return labels


def load_data(filename, response=None, selected_genes=None):
    logging.info("loading data from %s," % filename)
    if filename in cached_data:
        logging.info("loading from memory cached_data")
        data = cached_data[filename]
    else:
        data = pd.read_csv(filename, index_col=0)
        cached_data[filename] = data
    logging.info(data.shape)

    if response is None:
        if "response" in cached_data:
            logging.info("loading from memory cached_data")
            labels = cached_data["response"]
        else:
            raise ValueError(
                "abort: must read response first, but can't find it in cached_data"
            )
    else:
        labels = copy.deepcopy(response)

    # join with the labels
    all = data.join(labels, how="inner")
    all = all[~all["response"].isnull()]

    response = all["response"]
    samples = all.index

    del all["response"]
    x = all
    genes = all.columns

    if not selected_genes is None:
        intersect = set.intersection(set(genes), selected_genes)
        if len(intersect) < len(selected_genes):
            # raise Exception('wrong gene')
            logging.warning("some genes dont exist in the original data set")
        x = x.loc[:, intersect]
        genes = intersect
    logging.info(
        "loaded data %d samples, %d variables, %d responses "
        % (x.shape[0], x.shape[1], response.shape[0])
    )
    logging.info(len(genes))
    return x, response, samples, genes


# complete_features: make sure all the data_types have the same set of features_processing (genes)
def combine(
    x_list,
    y_list,
    rows_list,
    cols_list,
    data_type_list,
    combine_type,
    use_coding_genes_only=None,
):
    cols_list_set = [set(list(c)) for c in cols_list]

    if combine_type == "intersection":
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)
    logging.debug("step 1 union of gene features", len(cols))

    if use_coding_genes_only is not None:
        assert os.path.isfile(
            use_coding_genes_only
        ), "you specified a filepath to filter coding genes, but the file doesn't exist"
        f = os.path.join(use_coding_genes_only)
        coding_genes_df = pd.read_csv(f, sep="\t", header=None)
        coding_genes_df.columns = ["chr", "start", "end", "name"]
        coding_genes = set(coding_genes_df["name"].unique())
        cols = cols.intersection(coding_genes)
        logging.debug(
            "step 2 intersect w/ coding",
            len(coding_genes),
            "; coding AND in cols",
            len(cols),
        )

    # the unique (super) set of genes
    all_cols = list(cols)

    all_cols_df = pd.DataFrame(index=all_cols)

    df_list = []
    for x, y, r, c, d in zip(x_list, y_list, rows_list, cols_list, data_type_list):
        df = pd.DataFrame(x, columns=c, index=r)
        df = df.T.join(all_cols_df, how="right")
        df = df.T
        logging.info("step 3 fill NA-%s num NAs=" % d, df.isna().sum().sum())
        # IMPORTANT: using features in union will be filled zeros!!
        df = df.fillna(0)
        df_list.append(df)

    all_data = pd.concat(df_list, keys=data_type_list, join="inner", axis=1,)

    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)

    x = all_data.values
    # NOTE: only the last y is used; all else are discarded
    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y, how="left")

    y = y.values
    cols = all_data.columns
    rows = all_data.index
    logging.debug(
        "After combining, loaded data %d samples, %d variables, %d responses "
        % (x.shape[0], x.shape[1], y.shape[0])
    )

    return all_data, x, y, rows, cols


### For Reactome Pathways ###

def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def complete_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    #distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph


def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))

    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n)
            next = net.successors(n)
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
        layers.append(dict)
    return layers


class Reactome():

    def __init__(self, reactome_base_dir, relations_file_name, pathway_names_fn, pathway_genes_fn):
        self.reactome_base_dir =  reactome_base_dir
        self.relations_file_name = relations_file_name
        self.pathway_names_fn = pathway_names_fn
        self.pathway_genes_fn = pathway_genes_fn
        self.pathway_names = self.load_names()
        self.hierarchy = self.load_hierarchy()
        self.pathway_genes = self.load_genes()

    def load_names(self):
        filename = os.path.join(self.reactome_base_dir, self.pathway_names_fn)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['reactome_id', 'pathway_name', 'species']
        return df

    def load_genes(self):
        filename = os.path.join(self.reactome_base_dir, self.pathway_genes_fn)
        gmt = GMT()
        df = gmt.load_data(filename, pathway_col=1, genes_col=3)
        return df

    def load_hierarchy(self):
        filename = os.path.join(self.reactome_base_dir, self.relations_file_name)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['child', 'parent']
        return df


class ReactomeNetwork():

    def __init__(self, reactome_kws):
        self.reactome = Reactome(**reactome_kws)  # low level access to reactome pathways and genes
        self.netx = self.get_reactome_networkx()

    def get_terminals(self):
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes

    def get_roots(self):

        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    # get a DiGraph representation of the Reactome hierarchy
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
        net.name = 'reactome'

        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):

        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')

        return G

    def get_completed_network(self, n_levels):
        G = complete_network(self.netx, n_leveles=n_levels)
        return G

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_leveles=n_levels)
        return G

    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        # get the last layer (genes level)
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]  # set of terminal pathways
        # we need to find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            genes = genes_df[genes_df['group'] == pathway_name]['gene'].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes

        layers.append(dict)
        return layers


# data_dir = os.path.dirname(__file__)
class GMT():
    # genes_cols : start reading genes from genes_col(default 1, it can be 2 e.g. if an information col is added after the pathway col)
    # pathway col is considered to be the first column (0)
    def load_data(self, filename, genes_col=1, pathway_col=0):

        data_dict_list = []
        with open(filename) as gmt:

            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    dict = {'group': pathway, 'gene': gene}
                    data_dict_list.append(dict)

        df = pd.DataFrame(data_dict_list)
        # print df.head()

        return df

    def load_data_dict(self, filename):

        data_dict_list = []
        dict = {}
        with open(os.path.join(data_dir, filename)) as gmt:
            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.split('\t')
                dict[genes[0]] = genes[2:]

        return dict

    def write_dict_to_file(self, dict, filename):
        lines = []
        with open(filename, 'w') as gmt:
            for k in dict:
                str1 = '	'.join(str(e) for e in dict[k])
                line = str(k) + '	' + str1 + '\n'
                lines.append(line)
            gmt.writelines(lines)
        return

    def __init__(self):

        return


### Biology Pathway layer to map ###

def get_layer_maps(reactome, genes, n_levels, direction, add_unk_genes, verbose=False):
    reactome_layers = reactome.get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        if verbose: print('layer #', i)
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        if verbose: print('filtered_map', filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        if verbose: print('filtered_map', filter_df.shape)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

        # UNK, add a node for genes without known reactome annotation
        if add_unk_genes:
            if verbose: print('UNK ')
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, 'UNK'] = 1
        ####

        filtered_map = filtered_map.fillna(0)
        if verbose: print('filtered_map', filter_df.shape)
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        if verbose:
            logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps


def get_map_from_layer(layer_dict):
    pathways = list(layer_dict.keys())
    print('pathways', len(pathways))
    genes = list(itertools.chain.from_iterable(list(layer_dict.values())))
    genes = list(np.unique(genes))
    print('genes', len(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in list(layer_dict.items()):
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    # for k, v in layer_dict.items():
    #     print k, v
    #     df.loc[k,v] = 1
    # df= df.fillna(0)
    return df.T


