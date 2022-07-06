import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool

from typing import Sequence


class GCNNet(nn.Module):
    """A network based on graph convolutional operators.
    
    This applies a couple of graph convolutional operators followed by an MLP. Graph
    convolutional operators basically average over the node attributes from a given node
    plus its neighboring nodes, with weights proportional to edge weights and inversely
    proportional to the square roots of the node degrees. The final node attributes are
    obtained by passing through a fully-connected linear layer. See
    `torch_geometric.nn.GCNConv` for full details.

    The result from the graph convolutional layers is passed through an MLP, with class
    predictions obtained by (log) softmax.

    :param output_intermediate: if true, the module outputs not only the final class
        prediction, but also:
            `x1`: the result after the graph convolutional layers, passed through a ReLU
            `x2`: the result from the global mean pooling (before dropout)
    :param num_classes: number of output classes
    :param dims: dimensions of the input layers (`dims[0]`) and the various three hidden
        layers; should have length 4
    """

    def __init__(
        self,
        num_classes: int = 2,
        dims: Sequence = (128, 128, 64, 128),
        output_intermediate: bool = False,
    ):
        assert len(dims) == 4

        super(GCNNet, self).__init__()

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

        self.output_intermediate = output_intermediate

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()

        # dimension stays 128
        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        # x = F.dropout(x, p=0.5, training=self.training)

        # dimension goes down to 64
        x1 = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        # x1 = F.dropout(x1, p=0.5, training=self.training)

        # global pooling leads us into non-graph neural net territory
        x2 = global_mean_pool(x1, data.batch)
        x = F.dropout(x2, p=0.5, training=self.training)

        # back to 128-dimensions, then down to the number of classes
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # (log) softmax for class predictions
        y = self.m(x)

        # if asked to, return some intermediate results
        if self.output_intermediate:
            return y, x1, x2
        else:
            return y
