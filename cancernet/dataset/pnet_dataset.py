import os
import time
import copy
import gzip
import logging
import pickle
import json
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce


cached_data = {}  # all data read will be referenced here

class BrainDataSet(Dataset):
    """ Dataset for TCGA database for LGG and GBM brain cancers."""
    def __init__(self,data_path,response_path,gene_path,val_split=0.08,test_split=0.08,seed=19988):
        """  
        We use 3 features for each gene, one-hot encodings of genetic mutation, copy
        number amplification, and copy number deletion.
        data vector, x, is in the shape [patient, gene, feature]
        """

        self.data_path=data_path
        self.response_path=response_path
        self.gene_path=gene_path
        self.val_split=val_split
        self.test_split=test_split
        self.seed=seed
        ## Set numpy seed
        np.random.seed(seed)
        
        ## Load x data
        with open(self.data_path, 'rb') as fp:
            self.x = pickle.load(fp)
        self.x=torch.tensor(self.x,dtype=torch.float32)
            
        ## Load y data
        response_table=pd.read_csv(self.response_path)
        self.y=torch.tensor(response_table.values[0][1:],dtype=torch.float32).unsqueeze(1)
        
        ## Load genes
        with open(gene_path, 'r') as f:
            self.genes = json.load(f)
            
        ## Set split indices
        self._get_split_indices()
        
    def _get_split_indices(self):
        """ Get randomly drawn train, valid, test splits """
        all_idx=np.arange(len(self.y))
        np.random.shuffle(all_idx)
        num_valid=int(self.val_split*len(self.y))
        num_test=int(self.test_split*len(self.y))
        self.valid_idx=all_idx[0:num_valid]
        self.test_idx=all_idx[num_valid:num_valid+num_test]
        self.train_idx=all_idx[num_valid+num_test:]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx],self.y[idx]


class PnetDataSet(Dataset):
    """ Prostate cancer dataset, used to reproduce https://www.nature.com/articles/s41586-021-03922-4 """
    def __init__(
            self,
            num_features=3,
            root: Optional[str] = "./data/prostate/",
            valid_ratio: float = 0.102,
            test_ratio: float = 0.102,
            valid_seed: int = 0,
            test_seed: int = 7357,
        ):
        """  
        We use 3 features for each gene, one-hot encodings of genetic mutation, copy
        number amplification, and copy number deletion.
        data vector, x, is in the shape [patient, gene, feature]
        """

        self.num_features=num_features
        self.root=root
        self._files={}
        all_data,response=data_reader(filename_dict=self.raw_file_names,graph=False)
        self.subject_id=list(response.index)
        self.x=torch.tensor(all_data.to_numpy(),dtype=torch.float32)
        self.x=self.x.view(len(self.x),-1,self.num_features)
        self.y=torch.tensor(response.to_numpy(),dtype=torch.float32)

        self.genes=[g[0] for g in list(all_data.head(0))[0::self.num_features]]
        
        self.num_samples = len(self.y)
        self.num_test_samples = int(test_ratio * self.num_samples)
        self.num_valid_samples = int(valid_ratio * self.num_samples)
        self.num_train_samples = (
            self.num_samples - self.num_test_samples - self.num_valid_samples
        )
        self.split_index_by_rng(test_seed=test_seed, valid_seed=valid_seed)
        
    def split_index_by_rng(self, test_seed, valid_seed):
        """ Generate random splits for train, valid, test """
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
        """ Load train, valid, test splits from file """
        train_set = pd.read_csv(train_fp, index_col=0)
        valid_set = pd.read_csv(valid_fp, index_col=0)
        test_set = pd.read_csv(test_fp, index_col=0)
        
        patients_train=list(train_set.loc[:,"id"])
        both = set(self.subject_id).intersection(patients_train)
        self.train_idx=[self.subject_id.index(x) for x in both]
        
        patients_valid=list(valid_set.loc[:,"id"])
        both = set(self.subject_id).intersection(patients_valid)
        self.valid_idx=[self.subject_id.index(x) for x in both]
        
        patients_test=list(test_set.loc[:,"id"])
        both = set(self.subject_id).intersection(patients_test)
        self.test_idx=[self.subject_id.index(x) for x in both]
        
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
            f")"
        )

    @property
    def raw_file_names(self):
        return {
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
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx],self.y[idx]

class GraphDataSet(InMemoryDataset):
    """ PyG graph dataset to model genetic networks.
        Edge connections are imported from https://hb.flatironinstitute.org/ """
    def __init__(
        self,
        name="prostate_graph_humanbase",
        edge_tol=0.5,
        root: Optional[str] = "./data/prostate/",
        files: Optional[dict] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        valid_ratio: float = 0.102,
        test_ratio: float = 0.102,
        valid_seed: int = 0,
        test_seed: int = 7357,
    ):
        # the base-class constructor will generate the processed file if it is missing
        # self.all_data, self.response, self.edge_dict, self.edge_tol = all_data, response, edge_dict, edge_tol
        self.edge_tol = edge_tol
        self.name = name
        self._files = files or {}
        super().__init__(root, transform, pre_transform)

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


def data_reader(filename_dict,graph=True):
    # sanity checks for filename_dict
    assert "response" in filename_dict, "must parse a response file"
    fd = copy.deepcopy(filename_dict)
    # first get non-tumor genomic/config data types out
    if graph==True:
        ## Only check for graph files if we are loading graph data
        for f in filename_dict.values():
            if not os.path.isfile(f):
                raise FileNotFoundError(f)
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
        x_list.append(x)
        y_list.append(y)
        rows_list.append(info)
        cols_list.append(genes)
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
    if graph==True:
        return all_data, response, edge_dict
    else:
        return all_data, response


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
        intersect = list(set.intersection(set(genes), selected_genes))
        if len(intersect) < len(selected_genes):
            # raise Exception('wrong gene')
            logging.warning("some genes don't exist in the original data set")
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

    all_data = pd.concat(df_list, keys=data_type_list, join="inner", axis=1)

    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    # NOTE: sort this for reproducibility; FZZ 2022.10.12
    order = sorted(all_data.columns.levels[0])
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
