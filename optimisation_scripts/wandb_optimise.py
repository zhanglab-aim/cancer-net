import time
import os
import numpy as np

import torch, torch_geometric.transforms as T, torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

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

import wandb

from cancernet.arch import GATNet, InteractionNet, VisibleGraphInteractionNet
from cancernet.util import ProgressBar, InMemoryLogger, get_roc
from cancernet import PnetDataSet, ReactomeNetwork
from cancernet.dataset import get_layer_maps


def train():
    wandb.init(project="test-project", entity="cancer-net")

    dataset = PnetDataSet(
        root=os.path.join("../data", "prostate"),
        name="prostate_graph_humanbase",
        # files={'graph_file': "global.geneSymbol.gz"},
        edge_tol=0.5,
        pre_transform=T.Compose(
            [T.GCNNorm(add_self_loops=False), T.ToSparseTensor(remove_edge_index=False)]),)

    splits_root = os.path.join("/mnt/ceph/users/zzhang/cancer-net/data", "prostate", "splits")
    dataset.split_index_by_file(
        train_fp=os.path.join(splits_root, "training_set_0.csv"),
        valid_fp=os.path.join(splits_root, "validation_set.csv"),
        test_fp=os.path.join(splits_root, "test_set.csv"))

    pl.seed_everything(42, workers=True)

    n_epochs = 100
    batch_size = 10
    lr = wandb.config.lr


    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(dataset.train_idx),
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(dataset.valid_idx),
    )

    t0 = time.time()
    
    reactome_kws = dict(
        reactome_base_dir=os.path.join("/mnt/ceph/users/zzhang/cancer-net/data", "reactome"),
        relations_file_name="ReactomePathwaysRelation.txt",
        pathway_names_file_name="ReactomePathways.txt",
        pathway_genes_file_name="ReactomePathways.gmt",
    )
    reactome = ReactomeNetwork(reactome_kws)

    maps = get_layer_maps(
        genes=[g for g in dataset.node_index],
        reactome=reactome,
        n_levels=2,
        direction="root_to_leaf",
        add_unk_genes=False,
        verbose=True,
    )

    ##### MODELS #####
    #model = GATNet(dims=[3, 64, 256, 128], lr=lr)
    model = VisibleGraphInteractionNet(
        pathway_maps=maps,
        node_index=dataset.node_index,
        model_config={
            "inputs": 3,
            "outputs": 1,
            "hidden": wandb.config.hidden,
            "layers": wandb.config.layers,
        },
        sparse=False)

    logger = WandbLogger()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=50, verbose=False, mode="min")

    # XXX this cannot be fully deterministic on GPU because
    # XXX scatter_add_cuda_kernel does not have a deterministic implementation!
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=n_epochs,
        callbacks=[ProgressBar(), early_stop_callback],
        logger=logger,
        # deterministic=True
        )

    trainer.fit(model, train_loader, valid_loader)
    print(f"Training took {time.time() - t0:.1f} seconds.")


    fpr_train, tpr_train, train_auc, _, _ = get_roc(model, train_loader)
    fpr_valid, tpr_valid, valid_auc, _, _ = get_roc(model, valid_loader)
    
    fig, ax = plt.subplots()
    ax.plot(fpr_train, tpr_train, lw=2, label="train (area = %0.3f)" % train_auc)
    ax.plot(fpr_valid, tpr_valid, lw=2, label="validation (area = %0.3f)" % valid_auc)
    ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right", frameon=False)

    figure=wandb.Image(fig)
    wandb.log({"performance": figure})

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(dataset.test_idx),
        drop_last=True,
    )

    fpr_test, tpr_test, test_auc, ys, outs = get_roc(model, test_loader)
    test_acc=accuracy_score(ys, outs[:, 1] > 0.5)
    test_aupr=average_precision_score(ys, outs[:, 1])
    test_f1=f1_score(ys, outs[:, 1] > 0.5)
    test_precision=precision_score(ys, outs[:, 1] > 0.5)
    test_recall=recall_score(ys, outs[:, 1] > 0.5)

    wandb.run.summary["test acc"]=test_acc
    wandb.run.summary["test auc"]=test_auc,
    wandb.run.summary["test aupr"]=test_aupr,
    wandb.run.summary["test f1"]=test_f1,
    wandb.run.summary["test precision"]=test_precision,
    wandb.run.summary["test recall"]=test_recall

    print("accuracy", test_acc)
    print("auc", test_auc)
    print("aupr", test_aupr)
    print("f1", test_f1)
    print("precision", test_precision)
    print("recall", test_recall)
    wandb.finish()


sweep_configuration = {
    'method': 'bayes',
    'name': 'vgnn3',
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters':
    {
        'layers': {'max': 5,'min':2},
        'hidden': {'values': [8,12,16,20,24,32]},
        'lr': {'max': -2.30258509, 'min': -9.21034037,'distribution':'log_uniform'}
     }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project='vgnn3')

wandb.agent(sweep_id, function=train, count=30)
