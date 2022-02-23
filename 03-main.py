import os
import argparse
from types import SimpleNamespace

from typing import Iterable, Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel

import matplotlib.pyplot as plt

from TCGAData import TCGADataset

from arch.net import *


def train(
    model: torch.nn.Module,
    epoch: int,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    opt: SimpleNamespace,
) -> Tuple[float, float]:
    """Train a model.
    
    :param model: model to train
    :param epoch: current epoch number
    :param train_loader: loader for training data
    :param optimizer: PyTorch optimizer
    :param criterion: objective function
    :param opt: namespace of command-line arguments (needs `opt.lr` and `opt.parall`)
    :return: tuple `(avg_loss, avg_correct)`
    """
    model.train()

    # learning rate schedule
    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group["lr"] = opt.lr * 0.5

    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group["lr"] = opt.lr * 0.1

    total_loss = 0
    correct = 0
    total_samples = 0
    for data in train_loader:
        if not opt.parall:
            data = data.to(device)

        # run the model, collect the output
        optimizer.zero_grad()

        output, _, _ = model(data)
        output = output.squeeze()

        if opt.parall:
            y = torch.cat([d.y for d in data]).to(output.device)
        else:
            y = data.y

        # TT: GCN and GCN2 return different shapes, apparently?
        if len(output.shape) == 1:
            output = output.unsqueeze(0)

        # calculate loss, backpropagate
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # update accuracy counters
        pred = output.max(1)[1]
        correct += pred.eq(y).sum().item()
        total_loss += loss

        # keep track of total number of samples
        total_samples += len(y)

    # calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_correct = correct / total_samples

    # display progress
    print(f"Epoch: {epoch:02d}, Loss: {avg_loss:.4f}, Train Acc: {avg_correct:.4f}")

    return avg_loss, avg_correct


def test(model: torch.nn.Module, test_loader: Iterable, opt: SimpleNamespace) -> float:
    """Test a model.
    
    :param model: model to test
    :param test_loader: loader for test data
    :param opt: namespace of command-line arguments (`opt.parall`)
    :return: average correct predictions
    """
    model.eval()

    correct = 0
    total_samples = 0
    for data in test_loader:
        if not opt.parall:
            data = data.to(device)

        # run the model, collect the output
        output, _, _ = model(data)
        output = output.squeeze()

        if opt.parall:
            y = torch.cat([d.y for d in data]).to(output.device)
        else:
            y = data.y

        # update accuracy counter
        pred = output.max(1)[1]
        correct += pred.eq(y).sum().item()

        # keep track of total number of samples
        total_samples += len(y)

    # calculate average
    avg_correct = correct / total_samples
    return avg_correct


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset choice: 'brain', 'kidney', or 'lung'")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--stepsize", type=int, default=20, help="scheduler step size")
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="scheduler shrinking rate"
    )
    parser.add_argument(
        "--weightdecay", type=float, default=5e-2, help="regularization"
    )
    parser.add_argument("--arch", type=str, default="GCN", help="GCN or GCN2")
    parser.add_argument("--gpu", dest="gpu", default=True, action="store_true")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--parall", dest="parall", action="store_true")
    parser.add_argument("--batch", type=int, default=10, help="batch size")
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument("-seed", type=int, default=0, help="random seed")
    opt = parser.parse_args()
    print(opt)

    # choose device
    if opt.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("GPU not available, falling back to CPU")
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ensure reproducibility
    torch.manual_seed(opt.seed)

    # define conversion from string to numeric labels for each dataset
    # TT: maybe these would be better stored in a file?
    all_label_mappings = {
        # GBM: glioblastoma and patients die within a few months
        # LGG: low grade glioma and is assumed to be much more benign
        "brain": {"GBM": 1, "LGG": 0,},
        # XXX these are guessed based on slides, should double check!!
        "kidney": {"KICH": 1, "KIRC": 1, "KIRP": 0,},
        # XXX these are guessed based on slides, should double check!!
        "lung": {"LUAD": 1, "LUSC": 0,},
    }

    # load the data
    dataroot = os.path.join("data", opt.dataset)
    common_kwargs = {"root": dataroot, "label_mapping": all_label_mappings[opt.dataset]}
    if opt.arch == "GCN2":
        # GCN2 requires data to be in a sparse format
        pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
        dataset = TCGADataset(
            **common_kwargs, pre_transform=pre_transform, suffix="sparse"
        )
    elif opt.arch == "GCN":
        dataset = TCGADataset(**common_kwargs)
    else:
        raise ValueError(f"Unknown architecture: {opt.arch}.")
    print(dataset)

    # make and save figure with number of mutated genes per sample
    num_nodes_all = []
    for i, data in enumerate(dataset):
        num_nodes_all.append(data.x.shape[0])

    fig, ax = plt.subplots(constrained_layout=True)
    bin_max = 10 ** (np.ceil(np.log10(np.max(num_nodes_all))))
    ax.hist(num_nodes_all, bins=np.geomspace(1, bin_max, 20))
    ax.set_xlabel("number of mutated genes")
    ax.set_xscale("log")
    fig.savefig(os.path.join("figures", "num_nodes.png"))

    # perform a train / test split
    shuffled_indices = torch.randperm(len(dataset))
    n_training = 2 * len(dataset) // 3
    train_indices = shuffled_indices[:n_training]
    test_indices = shuffled_indices[n_training:]

    # if parall -->
    # train_loader = DataListLoader(dataset, batch_size=opt.batch, sampler=SubsetRandomSampler(train_indices),drop_last=True)
    # test_loader = DataListLoader(dataset, batch_size=opt.batch, sampler=SubsetRandomSampler(test_indices),drop_last=True)
    train_loader = DataLoader(
        dataset,
        batch_size=opt.batch,
        sampler=SubsetRandomSampler(train_indices),
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=opt.batch,
        sampler=SubsetRandomSampler(test_indices),
        drop_last=True,
    )

    # create the neural net
    if opt.arch == "GCN2":
        model = GCN2Net(
            hidden_channels=2048,
            num_layers=4,
            alpha=0.5,
            theta=1.0,
            shared_weights=False,
            dropout=0.2,
            flag=True,
        )
    elif opt.arch == "GCN":
        model = GCNNet(flag=True)
    else:
        raise ValueError(f"Unknown architecture: {opt.arch}.")

    print(model)

    # ensure we're using the right device, and use parallel data if asked to
    if opt.parall:
        model = DataParallel(model)
    model = model.to(device)

    # set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = F.nll_loss

    # train and test
    train_losses = []
    train_acces = []
    test_acces = []
    for epoch in range(1, opt.n_epochs + 1):
        train_loss, train_acc = train(
            model, epoch, train_loader, optimizer, criterion, opt
        )
        test_acc = test(model, test_loader, opt)

        train_losses.append(train_loss.cpu().detach().numpy())
        train_acces.append(train_acc)
        test_acces.append(test_acc)

        print("Test Acc: {:.4f}".format(test_acc))

    # save training and test learning curves
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(train_acces, label="train acc", linewidth=3)
    ax.plot(test_acces, label="test acc", linewidth=3)
    ax.plot(train_losses, "k--", label="train loss", linewidth=3)
    ax.legend(prop={"size": 16}, frameon=False)
    ax.set_xlabel("epoch", fontsize=16)
    fig.savefig(os.path.join("figures", "training.png"))

    print("finished")
