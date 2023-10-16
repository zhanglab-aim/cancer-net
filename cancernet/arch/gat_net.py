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
    def __init__(
        self,
        num_classes: int = 2,
        dims: Sequence = (128, 128, 64, 128),
        lr: float = 0.01,
    ):
        super().__init__(lr=lr)

        assert len(dims) == 4

        # GCNConv basically averages over the node attributes of a node's neighbors,
        # weighting by edge weights (if given), and including the node itself in the
        # average (i.e., including a self-loop edge with weight 1). The average is also
        # weighted by the product of the square roots of the node degrees (including the
        # self-loops), and is finally transformed by a learnable linear layer.
        self.prop1 = GCNConv(in_channels=dims[0], out_channels=dims[1])
        self.prop2 = GCNConv(in_channels=dims[1], out_channels=dims[2])

        self.fc1 = nn.Linear(dims[2], dims[3])
        self.fc2 = nn.Linear(dims[3], num_classes)
        self.m = nn.LogSoftmax(dim=1)

        self.gate_nn = nn.Sequential(
            nn.Linear(dims[2], 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.pool = GlobalAttention(gate_nn=self.gate_nn)

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()

        # dimension stays 128
        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        # x = F.dropout(x, p=0.5, training=self.training)

        # dimension goes down to 64
        x1 = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        # x1 = F.dropout(x1, p=0.5, training=self.training)

        # global pooling leads us into non-graph neural net territory
        # x2 = global_mean_pool(x1, data.batch)
        x2 = self.pool(x1, data.batch)
        x = F.dropout(x2, p=0.1, training=self.training)

        # back to 128-dimensions, then down to the number of classes
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # (log) softmax for class predictions
        y = self.m(x)

        # if asked to, return some intermediate results
        return y

    def configure_optimizers(self) -> Tuple[List, List]:
        """Set up optimizers and schedulers.

        This adds a simple scheduler.
        """
        optimizer_list, _ = super().configure_optimizers()

        lr_lambda = lambda epoch: 1.0 if epoch < 30 else 0.5 if epoch < 60 else 0.1
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer_list[0], lr_lambda=lr_lambda
        )

        return optimizer_list, [scheduler]


class GATNet(BaseNet):
    """ Network based on Graph Attention Networks https://arxiv.org/abs/1710.10903 
        Similar to a GCN, but each graph convolution consists of a self attention step.

        :param in_channels: number of input features
        :param hidden_channels: size of the latent node representations
        :param num_layers: number of message passing steps/graph convolutions
        :param heads: number of attention heads at each graph conv
        :param dropout: dropout rate applied at each hidden layer
        :param lr: learning rate
        """
    def __init__(self,in_channels: int=3,hidden_channels: int=128,num_layers:int=1,heads:int=1,dropout:float=0.1,lr:float=0.01):
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
    