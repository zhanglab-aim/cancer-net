import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool

from typing import Sequence

from cancernet.arch.base_net import BaseNet


class GCNNet(BaseNet):
    """A network based on graph convolutional operators.

    This applies a couple of graph convolutional operators followed by an MLP. Graph
    convolutional operators basically average over the node attributes from a given node
    plus its neighboring nodes, with weights proportional to edge weights and inversely
    proportional to the square roots of the node degrees. The final node attributes are
    obtained by passing through a fully-connected linear layer. See
    `torch_geometric.nn.GCNConv` for full details.

    The result from the graph convolutional layers is passed through an MLP, with class
    predictions obtained by (log) softmax.

    :param num_classes: number of output classes
    :param graph_dims: size of the latent node representations
    :param mlp_dims: size of the intermediate mlp used before classification
    :param layers: number of message passing steps/graph convolutions
    :param lr: learning rate
    """

    def __init__(
        self,
        num_classes: int = 2,
        graph_dims: int = 128,
        mlp_dims: int = 128,
        layers: int = 2,
        lr: float = 0.01,
    ):

        super().__init__(lr=lr)
        self.layers=layers

        # GCNConv basically averages over the node attributes of a node's neighbors,
        # weighting by edge weights (if given), and including the node itself in the
        # average (i.e., including a self-loop edge with weight 1). The average is also
        # weighted by the product of the square roots of the node degrees (including the
        # self-loops), and is finally transformed by a learnable linear layer.
        self.graphs = nn.ModuleList([GCNConv(in_channels=3, out_channels=graph_dims),nn.ReLU()])
        for aa in range(self.layers-1):
            self.graphs.append(GCNConv(in_channels=graph_dims, out_channels=graph_dims))
            self.graphs.append(nn.ReLU())
            
        self.fc1 = nn.Linear(graph_dims, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, num_classes)
        self.m = nn.LogSoftmax(dim=1)

    def forward(self, data):
        edge_attr = data.edge_attr.squeeze()

        x=data.x
        for aa in range(0,len(self.graphs),2):
            x = self.graphs[aa](x, data.edge_index, data.edge_attr)
            x = self.graphs[aa+1](x)

        # global pooling leads us into non-graph neural net territory
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        # (log) softmax for class predictions
        y = self.m(x)

        # if asked to, return some intermediate results
        return y
    