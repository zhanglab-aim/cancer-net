import os
import pickle
import pandas as pd
import numpy as np
import time
from tqdm import tqdm


from cancernet.arch import PNet
from cancernet import PnetDataSet, ReactomeNetwork
from cancernet.dataset import get_layer_maps
from cancernet.util import ProgressBar, InMemoryLogger, get_roc

import torch, torch_geometric.transforms as T, torch.nn.functional as F
from torch.utils.data.sampler import (
    SubsetRandomSampler,
    SequentialSampler,
    RandomSampler,
)
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

from typing import Iterable, Tuple

import argparse


parser = argparse.ArgumentParser(description="run P-Net for prostate cancer discovery")
parser.add_argument(
    "--n-hidden", required=True, type=int, help="number of hidden layers"
)
parser.add_argument(
    "--n-repeats",
    required=False,
    type=int,
    default=1,
    help="number of repeated measures",
)
args = parser.parse_args()

reactome_kws = dict(
    reactome_base_dir=os.path.join("/mnt/home/cpedersen/ceph/Data/data", "reactome"),
    relations_file_name="ReactomePathwaysRelation.txt",
    pathway_names_file_name="ReactomePathways.txt",
    pathway_genes_file_name="ReactomePathways.gmt",
)
reactome = ReactomeNetwork(reactome_kws)

prostate_root = os.path.join("/mnt/home/cpedersen/ceph/Data/data", "prostate")
dataset = PnetDataSet(
    root=prostate_root,
    name="prostate_graph_humanbase",
    edge_tol=0.5,
    pre_transform=T.Compose(
        [T.GCNNorm(add_self_loops=False), T.ToSparseTensor(remove_edge_index=False)]
    ),
)

# loads the train/valid/test split from pnet
splits_root = os.path.join(prostate_root, "splits")
dataset.split_index_by_file(
    train_fp=os.path.join(splits_root, "training_set_0.csv"),
    valid_fp=os.path.join(splits_root, "validation_set.csv"),
    test_fp=os.path.join(splits_root, "test_set.csv"),
)
pl.seed_everything(3478234589, workers=True)

n_epochs = 100
batch_size = 10
lr = 0.001
## Maps for pnet connections
maps = get_layer_maps(
    genes=[g for g in dataset.node_index],
    reactome=reactome,
    n_levels=args.n_hidden,
    direction="root_to_leaf",
    add_unk_genes=False,
    verbose=False,
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(dataset.train_idx),
    drop_last=True,
)
valid_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(dataset.valid_idx),
    drop_last=False,
)

test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(
        dataset.test_idx,
        generator=torch.Generator().manual_seed(43),
    ),
    drop_last=False,
)

results = []

for _ in tqdm(range(args.n_repeats)):
    model = PNet(
        layers=maps,
        num_genes=maps[0].shape[0],  # 9054
    )
    logger = InMemoryLogger()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="model_ckpts",
        monitor="val_loss",
        save_top_k=1,
        filename="bestmodel_pnet.torch",
        mode="min",
    )

    bestmodel_file = os.path.join("model_ckpts", "bestmodel_pnet.torch.ckpt")
    if os.path.isfile(bestmodel_file):
        os.remove(bestmodel_file)
    t0 = time.time()
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=n_epochs,
        callbacks=[ProgressBar(), early_stop_callback, checkpoint_callback],
        logger=logger,
    )
    trainer.fit(model, train_loader, valid_loader)
    print(f"Training took {time.time() - t0:.1f} seconds.")
    model = model.load_from_checkpoint(bestmodel_file)

    fpr_test, tpr_test, test_auc, ys, outs = get_roc(
        model, test_loader, exp=False, seed=None
    )
    test_acc = accuracy_score(ys, outs[:, 1] > 0.5)
    test_aupr = average_precision_score(ys, outs[:, 1])
    test_f1 = f1_score(ys, outs[:, 1] > 0.5)
    test_precision = precision_score(ys, outs[:, 1] > 0.5)
    test_recall = recall_score(ys, outs[:, 1] > 0.5)

    result = {}
    result["auc"] = test_auc
    result["accuracy"] = test_acc
    result["aupr"] = test_aupr
    result["f1"] = test_f1
    result["precision"] = test_precision
    result["recall"] = test_recall

    for k, v in result.items():
        print(k, v)

    # result['pred'] = pd.DataFrame(np.vstack([np.array(y_hat).squeeze(), ys]).T, index=info_test_, columns=['pred_%i'%i for i in range(len(y_hat))] + ['y'])
    # need help fixing here to keep consistent with TF
    result["pred"] = outs
    result["y"] = ys
    results.append(result)

    with open(f"pnet_results.h{args.n_hidden}.torch.pkl", "wb") as f:
        pickle.dump(results, f)

    torch.cuda.empty_cache()
