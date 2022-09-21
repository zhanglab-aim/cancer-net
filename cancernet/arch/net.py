import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

from cancernet.arch.base_net import BaseNet


class Net(BaseNet):
    """A neural net based on edge-conditioned convolutions.
    
    This uses two edge-conditioned convolutions (`torch_geometric.nn.NNConv`), each
    followed by top-`k` pooling (`torch_geometric.nn.TopKPooling`), then uses global
    max and mean-pooling on the node attributes to generate features for an MLP that
    ultimately performs binary classification.

    :param dim: dimensionality of input node attributes
    :param lr: learning rate
    """

    def __init__(self, dim: int = 128, lr: float = 0.01):
        super().__init__(lr=lr)

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
