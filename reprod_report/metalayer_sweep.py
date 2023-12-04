import time
import os
import numpy as np
import torch, torch_geometric.transforms as T, torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
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
import pickle

from cancernet.arch import MetaLayerNet
from cancernet.util import ProgressBar, InMemoryLogger, get_roc
from cancernet import PnetDataSet

project_string='hyperparam_sweeps'

def train():
    wandb.init(project=project_string)
    ## Import hyperparameters
    lr=wandb.config.lr
    hidden=wandb.config.hidden
    layers=wandb.config.layers
    print("lr=",lr)
    print("hidden=",hidden)
    print("layers=",layers)
    
    print(wandb.config)

    ## path to data
    base_data_string="../data"

    dataset = PnetDataSet(
        root=os.path.join(base_data_string, "prostate"),
        name="prostate_graph_humanbase",
        # files={'graph_file': "global.geneSymbol.gz"},
        edge_tol=0.5,
        pre_transform=T.Compose(
            [T.GCNNorm(add_self_loops=False), T.ToSparseTensor(remove_edge_index=False)]),)

    splits_root = os.path.join(base_data_string, "prostate", "splits")
    dataset.split_index_by_file(
        train_fp=os.path.join(splits_root, "training_set_0.csv"),
        valid_fp=os.path.join(splits_root, "validation_set.csv"),
        test_fp=os.path.join(splits_root, "test_set.csv"))

    pl.seed_everything(42, workers=True)

    n_epochs = 100
    batch_size = 10


    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(dataset.train_idx),
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(dataset.valid_idx),
        generator=torch.Generator().manual_seed(42),
    )

    t0 = time.time()

    ##### MODELS #####
    model = MetaLayerNet(layers=layers,
                            hidden=hidden,
                            lr=lr)
    print(model)

    n_param=sum(p.numel() for p in model.parameters())

    logger = WandbLogger()

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=n_epochs,
        logger=logger,
        enable_progress_bar = False
        )

    trainer.fit(model, train_loader, valid_loader)
    print(f"Training took {time.time() - t0:.1f} seconds.")

    fpr_train, tpr_train, train_auc, ys_train, outs_train = get_roc(model, train_loader)
    fpr_valid, tpr_valid, valid_auc, ys_valid, outs_valid = get_roc(model, valid_loader)

    train_acc=accuracy_score(ys_train, outs_train[:, 1] > 0.5)
    train_aupr=average_precision_score(ys_train, outs_train[:, 1])
    train_f1=f1_score(ys_train, outs_train[:, 1] > 0.5)
    train_precision=precision_score(ys_train, outs_train[:, 1] > 0.5)
    train_recall=recall_score(ys_train, outs_train[:, 1] > 0.5)
    
    valid_acc=accuracy_score(ys_valid, outs_valid[:, 1] > 0.5)
    valid_aupr=average_precision_score(ys_valid, outs_valid[:, 1])
    valid_f1=f1_score(ys_valid, outs_valid[:, 1] > 0.5)
    valid_precision=precision_score(ys_valid, outs_valid[:, 1] > 0.5)
    valid_recall=recall_score(ys_valid, outs_valid[:, 1] > 0.5)

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
        generator=torch.Generator().manual_seed(43),
    )

    fpr_test, tpr_test, test_auc, ys, outs = get_roc(model, test_loader)
    test_acc=accuracy_score(ys, outs[:, 1] > 0.5)
    test_aupr=average_precision_score(ys, outs[:, 1])
    test_f1=f1_score(ys, outs[:, 1] > 0.5)
    test_precision=precision_score(ys, outs[:, 1] > 0.5)
    test_recall=recall_score(ys, outs[:, 1] > 0.5)

    wandb.run.summary["valid acc"]=valid_acc
    wandb.run.summary["valid auc"]=valid_auc
    wandb.run.summary["valid aupr"]=valid_aupr
    wandb.run.summary["valid f1"]=valid_f1
    wandb.run.summary["valid precision"]=valid_precision
    wandb.run.summary["valid recall"]=valid_recall

    wandb.run.summary["test acc"]=test_acc
    wandb.run.summary["test auc"]=test_auc
    wandb.run.summary["test aupr"]=test_aupr
    wandb.run.summary["test f1"]=test_f1
    wandb.run.summary["test precision"]=test_precision
    wandb.run.summary["test recall"]=test_recall
    wandb.run.summary["test recall"]=test_recall

    wandb.run.summary["num parameters"]=n_param

    wandb.run.summary["save directory"]=wandb.run.dir

    torch.save(model.state_dict(), wandb.run.dir+"/model_weights.pt")

    results_dict={"num parameters":n_param,
                "lr":lr,
                "hidden":hidden,
                "layers":layers,

                "train truth": ys_train,
                "train predictions": outs_train[:, 1],
                "train acc": train_acc,
                "train auc": train_auc,
                "train aupr": train_aupr,
                "train f1": train_f1,
                "train precision": train_precision,
                "train recall": train_recall,

                "valid truth": ys_valid,
                "valid predictions": outs_valid[:, 1],
                "valid acc": valid_acc,
                "valid auc": valid_auc,
                "valid aupr": valid_aupr,
                "valid f1": valid_f1,
                "valid precision": valid_precision,
                "valid recall": valid_recall,

                "test truth":ys,
                "test predictions":outs[:, 1],
                "test acc":test_acc,
                "test auc":test_auc,
                "test aupr":test_aupr,
                "test f1": test_f1,
                "test precision": test_precision,
                "test recall": test_recall}

    ## Saves results as pickle file in wandb run folder
    with open(wandb.run.dir+'/results_dict.p', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return

        
sweep_configuration = {
    'method': 'random',
    'name': 'MetaLayer',
    'metric': {'goal': 'maximize', 'name': 'valid aupr'},
    'parameters':
    {
        'hidden': {'values': [32,64,128,256]},
        'lr': {'max': 0.1, 'min': 0.00001,'distribution':'log_uniform_values'},
        'layers': {'values': [1,2,3,4,5,6]},
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_string)

wandb.agent(sweep_id, function=train, count=50)
