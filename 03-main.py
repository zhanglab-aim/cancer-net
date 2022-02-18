import os
import numpy as np
import argparse
import time
import copy

import matplotlib.pyplot as plt
import deepdish as dd
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from os import listdir
from os.path import isfile, join
import h5py

from torch.utils.data.sampler import SubsetRandomSampler

from TCGAData import TCGADataset
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import NNConv, graclus, max_pool, max_pool_x, global_mean_pool
from torch_geometric.nn import GraphConv, TopKPooling, dense_diff_pool, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader
from utils import GNNExplainer

from arch.net import *

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--stepsize", type=int, default=20, help="scheduler step size")
parser.add_argument("--gamma", type=float, default=0.5, help="scheduler shrinking rate")
parser.add_argument("--weightdecay", type=float, default=5e-2, help="regularization")
parser.add_argument("--arch", type=str, default="GCN", help="GCN or GCN2")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--no-gpu", dest="gpu", action="store_false")
parser.add_argument("--parall", dest="parall", action="store_true")
parser.add_argument("--explain", dest="explain", action="store_true")
parser.add_argument("--batch", type=int, default=10, help="batch size")
parser.set_defaults(feature=True)
opt = parser.parse_args()
print(opt)
flag = True
# ------ Read column names from file
# dataroot = '../../data/frank/embedded'
dataroot = os.path.join(os.path.dirname(__file__), "data")
samples = [
    f for f in listdir(join(dataroot, "raw")) if isfile(join(dataroot, "raw", f))
]

if opt.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print(device)

# we need to clean the processed folder when we change arch type
pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
if opt.arch == "GCN2":
    dataset = TCGADataset(root=dataroot, pre_transform=pre_transform)
elif opt.arch == "GCN":
    dataset = TCGADataset(root=dataroot)

single_node_samples = []
num_nodes_all = []
for i, data in enumerate(dataset):
    num_nodes_all.append(data.x.shape[0])
    if data.x.shape[0] <= 2 * torch.cuda.device_count():  # in case in parall gpus
        single_node_samples.append(i)

plt.figure()
plt.hist(np.array(num_nodes_all), bins=50)
plt.savefig("figures/num_nodes.png")

# TT: the masking procedure here was wrong -- `mask` should be type bool!
# mask = torch.ones(len(dataset), dtype=torch.long)
mask = torch.ones(len(dataset), dtype=torch.bool)
mask[single_node_samples] = 0
dataset = dataset[mask]


if opt.parall:
    train_indices = list(range(300)) + list(range(600, len(dataset)))
    test_indices = list(range(300, 600))
    # parall
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
    d = dataset
else:
    train_indices = list(range(300)) + list(range(600, len(dataset)))
    test_indices = list(range(300, 600))
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
    d = dataset


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


if opt.arch == "GCN2":
    model = GCN2Net(
        hidden_channels=2048,
        num_layers=4,
        alpha=0.5,
        theta=1.0,
        shared_weights=False,
        dropout=0.2,
        flag=flag,
    )
elif opt.arch == "GCN":
    model = GCNNet(flag=flag).to(device)

print(model)

if opt.parall:
    model = DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
if opt.arch == "GCN2":
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = F.nll_loss
elif opt.arch == "GCN":
    criterion = F.nll_loss


# optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)


def train(epoch):
    model.train()

    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group["lr"] = opt.lr * 0.5

    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group["lr"] = opt.lr * 0.1

    total_loss = 0
    correct = 0
    for data in train_loader:
        if not opt.parall:
            data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        if flag:
            fc_embed = output[2]
            node_embed = output[1]
            output = output[0]
        output = output.squeeze()

        if opt.parall:
            y = torch.cat([d.y for d in data]).to(output.device)
        else:
            y = data.y

        # if opt.arch == 'GCN2':
        #     loss = criterion(output, y.float())
        # elif opt.arch == 'GCN':
        #     if len(output.shape) == 1:
        #         output = output.unsqueeze(0)
        #     loss = criterion(output, y)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        loss = criterion(output, y)

        pred = output.max(1)[1]
        correct += pred.eq(y).sum().item()
        total_loss += loss
        loss.backward()
        optimizer.step()
    print(
        "Epoch: {:02d}, Loss: {:.4f}, Train Acc: {:.4f}".format(
            epoch, total_loss / len(train_loader), correct / len(train_indices)
        )
    )

    return total_loss / len(train_loader), correct / len(train_indices)


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        if not opt.parall:
            data = data.to(device)
        output = model(data)
        if flag:
            fc_embed = output[2]
            node_embed = output[1]
            output = output[0]
        output = output.squeeze()

        pred = output.max(1)[1]
        if opt.parall:
            y = torch.cat([d.y for d in data]).to(output.device)
        else:
            y = data.y

        correct += pred.eq(y).sum().item()
    return correct / len(test_indices)


train_losses = []
train_acces = []
test_acces = []
for epoch in range(1, 101):
    # if epoch==15:
    #     import pdb
    #     pdb.set_trace()
    train_loss, train_acc = train(epoch)
    test_acc = test()
    train_losses.append(train_loss.cpu().detach().numpy())
    train_acces.append(train_acc)
    test_acces.append(test_acc)
    print("Test Acc: {:.4f}".format(test_acc))

plt.figure()
plt.plot(train_acces, label="train acc", linewidth=3)
plt.plot(test_acces, label="test acc", linewidth=3)
plt.plot(train_losses, "k--", label="train loss", linewidth=3)
plt.legend(prop={"size": 16})
plt.xlabel("epoch", fontsize=16)
plt.savefig("training.png")
plt.show()


if opt.explain:
    # import pdb
    # pdb.set_trace()
    # explainer = GNNExplainer(model.module, epochs=200, return_type='log_prob')
    explainer = GNNExplainer(model, epochs=200, return_type="log_prob")
    node_idx = 10
    dataset_x = TCGADataset(root=dataroot)
    train_loader = DataLoader(
        dataset_x, batch_size=1, sampler=SubsetRandomSampler(train_indices)
    )
    for data in train_loader:
        data = data.to(device)
        node_feat_mask, edge_mask = explainer.explain_graph(data)
        plt.figure()
        plt.hist(edge_mask.detach().cpu().numpy(), bins=1000)
        plt.xlabel("edge mask")
        plt.ylabel("population")
        plt.savefig("figures/hist.png")

        th = np.percentile(edge_mask.detach().cpu().numpy(), 99.9)
        plt.figure(figsize=(50, 50))
        # import pdb
        # pdb.set_trace()
        # ax, G = explainer.visualize_subgraph(node_idx, data.edge_index, edge_mask, y=data.y, threshold=th)
        ax, G = explainer.visualize_subgraph(
            node_idx, data.edge_index, edge_mask, threshold=th
        )
        plt.savefig("figures/explain.png")
        # plt.show()
        break

print("finished")
