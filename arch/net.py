import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
from torch_geometric.nn import GraphConv, TopKPooling, dense_diff_pool, SAGPooling, GATConv,GCNConv,GCNConv, GCN2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import  DataParallel
from torch_geometric.data import DataListLoader
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool,global_sort_pool,dense_diff_pool
from torch_geometric.nn import PairNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 128
        n1 = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 64 * dim))
        self.conv1 = NNConv(dim, 64, n1,aggr='mean')
        self.pool1 = TopKPooling(64, ratio=0.5)
        n2 = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 64 * 64))
        self.conv2 = NNConv(64, 64, n2,aggr='mean')
        self.pool2 = TopKPooling(64, ratio=0.5)

        self.fc1 = torch.nn.Linear(128+128, 64)
        self.fc2 = torch.nn.Linear(64, 8)
        self.fc3 = torch.nn.Linear(8, 2)


    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index,edge_attr, batch,_,_ = self.pool1(x, edge_index,edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _,_ = self.pool2(x, edge_index,edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = x1 + x2 + x3
        x = torch.cat([x1,x2], dim=1)
        # x = x2

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x

dims = [128, 128, 64, 64, 128, 128, 256]
class GCNNet(torch.nn.Module):
    def __init__(self,flag):
        dim = 128
        super(GCNNet, self).__init__()
        self.prop1 = GCNConv(in_channels = dim, out_channels = dims[1])
        self.prop2 = GCNConv(in_channels = dims[1], out_channels = dims[2])
        # self.prop3 = GCNConv(in_channels = dims[2], out_channels = dims[3])
        # self.prop4 = GCNConv(in_channels = dims[3], out_channels = dims[4])
        self.fc1 = Linear(dims[2], dims[5])
        # self.fc2 = Linear(dims[5], dims[6])
        # self.fc3 = Linear(dims[6], dims[2])
        self.fc2 = Linear(dims[5], 2)
        self.m = nn.LogSoftmax(dim=1)
        self.flag = flag

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()
        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        # x1 = F.dropout(x1, p=0.5, training=self.training)
        # x = F.relu(self.prop3(x, data.edge_index, data.edge_attr))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.prop4(x, data.edge_index, data.edge_attr))
        #x2 = global_sort_pool(x1, data.batch, 1)
        x2 = gap(x1, data.batch)
        x = F.dropout(x2, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.fc4(x))

        if self.flag == True:
            return self.m(x),x1,x2
        else:
            return self.m(x)


class GCN2Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0, flag = False):
        super(GCN2Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(128, hidden_channels))
        self.lins.append(Linear(hidden_channels * 2, hidden_channels//2))
        self.lins.append(Linear(hidden_channels//2, 2))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout
        self.pnorm = PairNorm()
        self.bn = nn.BatchNorm1d(hidden_channels//2)
        self.m = nn.LogSoftmax(dim=1)
        self.flag = flag

    def forward(self, data):
        x, adj_t, batch = data.x, data.adj_t, data.batch
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()
            x = self.pnorm(x)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = global_sort_pool(x, batch, 1)
        x1 = x
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.lins[1](x2)
        x = self.bn(x.relu())
        x = self.lins[2](x)

        if self.flag == True:
            return self.m(x), x1, x2
        else:
            return self.m(x)