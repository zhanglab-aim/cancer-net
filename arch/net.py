from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    NNConv,
    TopKPooling,
    GCNConv,
    GCNConv,
    GCN2Conv,
    PairNorm,
)
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention


class Net(torch.nn.Module):
    """A neural net based on edge-conditioned convolutions.
    
    This uses two edge-conditioned convolutions (`torch_geometric.nn.NNConv`), each
    followed by top-`k` pooling (`torch_geometric.nn.TopKPooling`), then uses global
    max and mean-pooling on the node attributes to generate features for an MLP that
    ultimately performs binary classification.

    :param dim: dimensionality of input node attributes
    """

    def __init__(self, dim: int = 128):
        super(Net, self).__init__()

        # NNConv uses an MLP to convert dim1-dimensional input node attributes into
        # dim2-dimensional output node attributes (here dim1=dim, dim2=64) then adds a
        # convolutional component to each node attribute.

        # The convolutional component is an average (because of aggr="mean" below) over
        # all neighbors of the dot product between the node's input attribute and a
        # matrix obtained by applying a neural network (here, that network is `n1`) to
        # the edge attribute. More specifically, the neural net returns a flattened
        # version of this matrix.

        # Edge attributes are 1d here, so the NN blows that up to 4d, applies relu, then
        # blows it up again to `64 * dim` dimensions.
        n1 = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 64 * dim))
        self.conv1 = NNConv(dim, 64, n1, aggr="mean")

        # TopKPooling learns a `dim`-dimensional weight vector that it projects each
        # node attribute on, normalizing by the L2 norm of the weight, and then passes
        # the result through `tanh`. Then it executes the top-k pooling operation
        # itself: selecting the `k` nodes with the largest projected values. The node
        # attributes are set to the original node attributes for the nodes that are
        # kept, multiplied by the tanh-transformed projected input node attributes.

        # Here `dim = 64`, and `ratio = 0.5`, so that the number of nodes is reduced to
        # half (more precisely, `ceil(ratio * N)`, where `N` is the number of nodes in
        # the input).
        self.pool1 = TopKPooling(64, ratio=0.5)

        n2 = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 64 * 64))
        self.conv2 = NNConv(64, 64, n2, aggr="mean")
        self.pool2 = TopKPooling(64, ratio=0.5)

        self.fc1 = torch.nn.Linear(128 + 128, 64)
        self.fc2 = torch.nn.Linear(64, 8)
        self.fc3 = torch.nn.Linear(8, 2)

    def forward(self, data):
        x, edge_index, batch, edge_attr = (
            data.x,
            data.edge_index,
            data.batch,
            data.edge_attr,
        )

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x, edge_index, edge_attr, batch
        )

        # `gobal_max_pool` and `global_mean_pool` calculate either the (component-wise)
        # maximum or the mean of the node attributes, where max or mean are taken over
        # all nodes in the graph. No averaging is done across batches (corresponding to
        # subjects here)
        # x1 is 64 + 64 = 128-dimensional
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x, edge_index, edge_attr, batch
        )

        # x2 is 64 + 64 = 128-dimensional
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # concatenate the outputs from the two conv+pool layers -- I guess this counts
        # as a kind of skip connection
        x = torch.cat([x1, x2], dim=1)

        # reduces to 64 dimensions
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # down to 8 dimensions
        x = F.relu(self.fc2(x))

        # final linear layer reduces to 2 dimensions
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x

class GATNet(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dims: Sequence = (128, 128, 64, 128),
        output_intermediate: bool = False,
    ):
        assert len(dims) == 4

        super(GATNet, self).__init__()

        # GCNConv basically averages over the node attributes of a node's neighbors,
        # weighting by edge weights (if given), and including the node itself in the
        # average (i.e., including a self-loop edge with weight 1). The average is also
        # weighted by the product of the square roots of the node degrees (including the
        # self-loops), and is finally transformed by a learnable linear layer.
        self.prop1 = GCNConv(in_channels=dims[0], out_channels=dims[1])
        self.prop2 = GCNConv(in_channels=dims[1], out_channels=dims[2])

        self.fc1 = torch.nn.Linear(dims[2], dims[3])
        self.fc2 = torch.nn.Linear(dims[3], num_classes)
        self.m = nn.LogSoftmax(dim=1)
        
        self.gate_nn = nn.Sequential(nn.Linear(dims[2], 32), nn.ReLU(), nn.Linear(32, 1) )
        self.pool = GlobalAttention(gate_nn = self.gate_nn)

        self.output_intermediate = output_intermediate

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()

        # dimension stays 128
        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        #x = F.dropout(x, p=0.5, training=self.training)

        # dimension goes down to 64
        x1 = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        #x1 = F.dropout(x1, p=0.5, training=self.training)

        # global pooling leads us into non-graph neural net territory
        #x2 = global_mean_pool(x1, data.batch)
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
        if self.output_intermediate:
            return y, x1, x2
        else:
            return y


class GCNNet(torch.nn.Module):
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

        self.fc1 = torch.nn.Linear(dims[2], dims[3])
        self.fc2 = torch.nn.Linear(dims[3], num_classes)
        self.m = nn.LogSoftmax(dim=1)

        self.output_intermediate = output_intermediate

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()

        # dimension stays 128
        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        #x = F.dropout(x, p=0.5, training=self.training)

        # dimension goes down to 64
        x1 = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        #x1 = F.dropout(x1, p=0.5, training=self.training)

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


class GCN2Net(torch.nn.Module):
    """A much larger network based on extended graph convolutional operators.
    
    This uses a dropout followed by a fully-connected layer with ReLU to blow up the
    input node attributes to `hidden_channels`-dimensions, then passes the resulting
    graph through a series of graph convolutional operators with initial residual
    connections and identity mapping (GCNII). This basically averages attributes over
    neighboring nodes, like normal `GCN` (see `GCNNet`), but it includes "skip"
    connections to some "initial" representation, and it also "shrinks" the linear
    weights acting on top of the convolution result towards the identity, with stronger
    shrinkage for deeper layers. See `torch_geometric.nn.GCN2Conv` for full details.

    The convolutional operators are followed by pair normalization, which aims to avoid
    oversmoothing (see `torch_geometric.nn.PairNorm`). Dropout layers are used for
    regularization, one before each convolutional layer.

    The result from the graph convolutional layers is passed through an MLP, with class
    predictions obtained by (log) softmax. The MLP also includes a batchnorm layer.

    :param hidden_channels: dimensionality of the node attributes during the
        convolutional layers
    :param num_layers: number of `GCN2Conv` layers
    :param alpha: strength of initial connections in `GCN2Conv` layers
    :param theta: strength of identity mapping in `GCN2Conv` layers
    :param num_classes: number of output classes
    :param shared_weights: whether to use different weight matrices for the convolution
        result and the initial residual; see `torch_geometric.nn.GCN2Conv`
    :param dropout: dropout strength
    :param output_intermediate: if true, the module outputs not only the final class
        prediction, but also:
            `x1`: the result after the graph convolutional layers
            `x2`: the result from the global pooling
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        alpha: float,
        theta: float,
        dim: int = 128,
        num_classes: int = 2,
        shared_weights: bool = True,
        dropout: float = 0.0,
        output_intermediate: bool = False,
    ):
        super(GCN2Net, self).__init__()

        # ModuleList makes PyTorch aware of the parameters for each module in the list
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(dim, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels * 2, hidden_channels // 2))
        self.lins.append(torch.nn.Linear(hidden_channels // 2, num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            # GCN2Conv is a graph convolutional layer similar to GCNConv, where node
            # attributes are averaged over neighbors. There are two difference,
            # controlled by the parameters `alpha` and `theta`:
            #   * it has skip connections so that the result from the convolution is
            #     combined with an "initial feature representation" (which is passed to
            #     the `GCN2Conv` layer at call time), with weights `(1 - alpha)` and
            #     `alpha`
            #   * the fully-connected linear layer that is applied to each node
            #     attribute after the convolution operation is "shrunk" towards the
            #     identity by a layer-dependent amount `1 - beta`, which is related to
            #     the parameter `theta` below by `beta = log(theta / l + 1)`. Thus
            #     earlier layers exprience less shrinkage, while later layers are pulled
            #     close to the identity.
            # A `GCN2Conv` behaves just as a `GCNConv` when `alpha=0` and `theta=0`.
            self.convs.append(
                GCN2Conv(
                    hidden_channels,
                    alpha,
                    theta,
                    layer + 1,
                    shared_weights,
                    normalize=False,
                )
            )

        self.dropout = dropout

        # PairNorm is a normalization step meant to guard against excessive smoothing
        # from the graph convolutional layers. It centers each node attribute to the
        # mean across all nodes, and normalizes by the variance of all the node
        # attributes at all nodes (i.e., flattening over both nodes and components).
        self.pnorm = PairNorm()

        # batch normalization
        self.bn = nn.BatchNorm1d(hidden_channels // 2)
        self.m = nn.LogSoftmax(dim=1)
        self.output_intermediate = output_intermediate

    def forward(self, data):
        # XXX what is the point of doing ToSparseTensor transform in the pre-processing?
        #     GCN2Conv already adds self-loops, and how else is `adj_t` different from
        #     edge index?
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

        # building some intermediate results and doing global pooling
        x1 = x
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # switch to usual MLP (with batchnorm) to get the final answer
        x = self.lins[1](x2)
        x = self.bn(x.relu())

        x = self.lins[2](x)

        # (log) softmax for class predictions
        y = self.m(x)

        # if asked to, return intermediate results
        if self.output_intermediate == True:
            return y, x1, x2
        else:
            return y
