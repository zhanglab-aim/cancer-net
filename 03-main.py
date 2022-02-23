import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel

import matplotlib.pyplot as plt

from TCGAData import TCGADataset

# from utils import GNNExplainer
from arch.net import *


def train(model, epoch, train_loader, optimizer, criterion, opt):
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

        output, _, _ = model(data)
        output = output.squeeze()

        if opt.parall:
            y = torch.cat([d.y for d in data]).to(output.device)
        else:
            y = data.y

        # TT: GCN and GCN2 return different shapes?
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
            epoch, total_loss / len(train_loader), correct / len(train_loader.dataset)
        )
    )

    return total_loss / len(train_loader), correct / len(train_loader.dataset)


def test(model, test_loader, opt):
    model.eval()
    correct = 0

    for data in test_loader:
        if not opt.parall:
            data = data.to(device)
        output, _, _ = model(data)
        output = output.squeeze()

        pred = output.max(1)[1]
        if opt.parall:
            y = torch.cat([d.y for d in data]).to(output.device)
        else:
            y = data.y

        correct += pred.eq(y).sum().item()
    return correct / len(test_loader.dataset)


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
    parser.add_argument("--explain", dest="explain", action="store_true")
    parser.add_argument("--batch", type=int, default=10, help="batch size")
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="number of training epochs"
    )
    opt = parser.parse_args()
    print(opt)

    # read data file names
    dataroot = os.path.join("data", opt.dataset)
    with open(os.path.join(dataroot, "samples.txt"), "rt") as f:
        samples = [_.strip() for _ in f.readlines()]
        samples = [_ for _ in samples if len(_) > 0]

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

    # define conversion from string to numeric labels for each dataset
    # TT: maybe these would be better stored in a file?
    all_label_mappings = {
        "brain": {
            # GBM: glioblastoma and patients die within a few months
            # LGG: low grade glioma and is assumed to be much more benign
            b"GBM": 1,
            b"LGG": 0,
        },
        "kidney": {
            # XXX these are guessed based on slides, should double check!!
            b"KICH": 1,
            b"KIRC": 1,
            b"KIRP": 0,
        },
        "lung": {
            # XXX these are guessed based on slides, should double check!!
            b"LUAD": 1,
            b"LUSC": 0,
        },
    }

    # load the data
    common_kwargs = {
        "root": dataroot,
        "files": samples,
        "label_mapping": all_label_mappings[opt.dataset],
        "name": opt.dataset,
    }
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

    # identify samples where the number of nodes is not more than 2x the number of GPUs
    # (number of nodes = number of mutated genes)
    # TT: why?
    single_node_samples = []
    num_nodes_all = []
    for i, data in enumerate(dataset):
        num_nodes_all.append(data.x.shape[0])
        if data.x.shape[0] <= 2 * torch.cuda.device_count():  # in case in parall gpus
            single_node_samples.append(i)

    # save a histogram of number of nodes per sample
    # TT: not sure this makes sense here
    plt.figure()
    plt.hist(np.array(num_nodes_all), bins=50)
    plt.savefig("figures/num_nodes.png")

    # choose only datasets that don't fit on single nodes (according to def above)
    # TT: why?!
    mask = torch.ones(len(dataset), dtype=torch.bool)
    mask[single_node_samples] = 0
    dataset = dataset[mask]

    train_indices = list(range(300)) + list(range(600, len(dataset)))
    test_indices = list(range(300, 600))
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

    # train
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
    plt.figure()
    plt.plot(train_acces, label="train acc", linewidth=3)
    plt.plot(test_acces, label="test acc", linewidth=3)
    plt.plot(train_losses, "k--", label="train loss", linewidth=3)
    plt.legend(prop={"size": 16})
    plt.xlabel("epoch", fontsize=16)
    plt.savefig("training.png")

    # TT: not sure what this was supposed to do; it probably won't work without change
    # if opt.explain:
    #     explainer = GNNExplainer(model, epochs=200, return_type="log_prob")
    #     node_idx = 10
    #     dataset_x = TCGADataset(root=dataroot)
    #     train_loader = DataLoader(
    #         dataset_x, batch_size=1, sampler=SubsetRandomSampler(train_indices)
    #     )
    #     for data in train_loader:
    #         data = data.to(device)
    #         node_feat_mask, edge_mask = explainer.explain_graph(data)
    #         plt.figure()
    #         plt.hist(edge_mask.detach().cpu().numpy(), bins=1000)
    #         plt.xlabel("edge mask")
    #         plt.ylabel("population")
    #         plt.savefig("figures/hist.png")

    #         th = np.percentile(edge_mask.detach().cpu().numpy(), 99.9)
    #         plt.figure(figsize=(50, 50))
    #         ax, G = explainer.visualize_subgraph(
    #             node_idx, data.edge_index, edge_mask, threshold=th
    #         )
    #         plt.savefig("figures/explain.png")
    #         break

    print("finished")
