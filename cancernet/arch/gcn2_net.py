import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCN2Conv, PairNorm, global_max_pool, global_mean_pool

from cancernet.arch.base_net import BaseNet


class GCN2Net(BaseNet):
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
    :param lr: learning rate
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
        lr: float = 0.01,
    ):
        super().__init__(lr=lr)

        # ModuleList makes PyTorch aware of the parameters for each module in the list
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(dim, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels * 2, hidden_channels // 2))
        self.lins.append(nn.Linear(hidden_channels // 2, num_classes))

        self.convs = nn.ModuleList()
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
        return y
