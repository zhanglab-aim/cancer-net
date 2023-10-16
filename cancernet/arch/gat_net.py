import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import GCNConv, GlobalAttention
from torch_geometric.nn.conv import GATConv

from typing import Sequence

from cancernet.arch.base_net import BaseNet
from typing import Tuple, List


class GATNet(BaseNet):
    """ Network based on Graph Attention Networks https://arxiv.org/abs/1710.10903 
        Similar to a GCN, but each graph convolution consists of an attention step.

        :param in_channels: number of input features
        :param hidden_channels: size of the latent node representations
        :param num_layers: number of message passing steps/graph convolutions
        :param heads: number of attention heads at each graph conv
        :param dropout: dropout rate applied at each hidden layer
        :param lr: learning rate
        """
    def __init__(self,in_channels: int=3,hidden_channels: int=128,num_layers: int=1,heads: int=1,dropout: float=0.1,lr: float=0.01):
        super().__init__(lr=lr)
        self.num_layers=num_layers
        self.graphs = nn.ModuleList([GATConv(in_channels=in_channels,out_channels=hidden_channels,heads=heads,dropout=0.1,concat=False)])
        for aa in range(self.num_layers-1):
            self.graphs.append(GATConv(in_channels=hidden_channels,out_channels=hidden_channels,heads=heads,dropout=0.1,concat=False))
        
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.pool = GlobalAttention(gate_nn=self.gate_nn)
        self.fc1 = nn.Linear(hidden_channels, 32)
        self.fc2 = nn.Linear(32, 2)
        self.m = nn.LogSoftmax(dim=1)
    
    def forward(self,data):
        x=self.graphs[0](data.x,data.edge_index,edge_attr=data.edge_attr)
        for aa in range(self.num_layers-1):
            x=self.graphs[aa+1](x,data.edge_index,edge_attr=data.edge_attr)
        x = self.pool(x, data.batch)
        x = F.dropout(x, p=0.1, training=self.training)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)

        # (log) softmax for class predictions
        return self.m(x)
    