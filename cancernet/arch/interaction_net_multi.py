"""A metalayer graph net."""

import torch
import numpy as np
import pandas as pd
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

from torch_geometric.nn import MetaLayer
from torch_geometric.utils import subgraph

from cancernet.arch.base_net import BaseNet
from cancernet.util import scatter_nd
from typing import Tuple, List


class EdgeModel(torch.nn.Module):
    def __init__(self, node_size: int, edge_attr_size: int, hidden: int):
        """Initialize the edge model.

        :param node_size: TODO fill
        :param edge_attr_size: TODO fill
        :param hidden: TODO fill
        """
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(node_size * 2 + edge_attr_size, hidden),
            BatchNorm1d(hidden),
            ReLU(),
            Linear(hidden, hidden),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        """Update edge attributes.

        :param src: node features of the sending node; shape `(E, F_x)` (`E` = number of
            edges, `F_x` number of node features)
        :param dest: node features for the "receiving" nodes; shape `(E, F_x)`
        :param edge_attr: edge features; shape `(E, F_e)` (`F_e` number of edge features)
        :param u: global features, currently unused
        :param batch: batch index for each edge; shape `(E,)`; max entry `B - 1`, where
            `B` is the batch size
        :return: [E, F_h] updated edge features after a "message pass" step; shape
            `(E, F_h)` (`F_h` = size of hidden layers)
        """
        if len(edge_attr.shape) == 1:
            out = torch.cat([src, dest, edge_attr.reshape(-1, 1)], 1)
        else:
            out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden: int):
        """Initialize the node model.

        :param input_size: TODO fill
        :param hidden: TODO fill
        """
        super().__init__()
        self.message_function = Sequential(
            Linear(input_size + hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
        )
        self.node_mlp = Sequential(
            Linear(input_size + hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """Update node attributes.

        Takes node features & edge features, updates node features based on the features
        of the sending and receiving node, and edge features of each connection.

        :param x: node features; shape `(N, F_x)` (`N` = number of nodes, `F_x` = number
            of node features); NB: `F_x` can be different for different layers of the
            graph (i.e. the input feature size is currently 3, while the latent node
            feature size is of size `hidden`)
        :param edge_index: list of indices describing the sending and receiving nodes of
            each edge; shape `(2, E)` (`E` = number of edges)
        :param edge_attr: edge feature; shape `(E, F_e)` (`F_e` = number of edge
            features); NB: this can also be different for different layers
        :param u: global features; currently unused
        :param batch: batch index for each edge; shape `(E,)`; max entry `B - 1`, where
            `B` is the batch size
        :returns: tensor of shape `(N, F_h)` (`F_h` = size of hidden layers)
        """
        send_idx, rec_idx = edge_index

        # tensor of node features of sending nodes, concatenated with the edge features
        out = torch.cat([x[send_idx], edge_attr], dim=1)
        out = self.message_function(out)
        # Aggregation step - for each receiving node, take the average of the hidden
        # layer outputs connected to that node
        out = scatter_add(out, rec_idx, dim=0, dim_size=x.size(0))
        # Finally concat each node feature with the hidden layer outputs, pass to one
        # final MLP
        return self.node_mlp(torch.cat([x, out], dim=1))


class GlobalModel(torch.nn.Module):
    def __init__(self, hidden: int, outputs: int):
        super().__init__()
        self.global_mlp = Sequential(
            Linear(hidden, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, outputs)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_mean(x, batch, dim=0)
        return self.global_mlp(out)


class InteractionNetworkMulti(BaseNet):
    """Class to stack multiple MetaLayers."""

    def __init__(self, layers: int, hidden: int, lr: float = 0.001):
        """Initialize the stack of metalayers.

        The general procedure for the `MetaLayer` is as follows.
            1. The `EdgeModel` takes the node features and edge features. For each edge
               connection, the node and edge features for each edge are concatenated
               together and passed to an MLP. The output is then a set of updated node
               features.
            2. The `NodeModel` takes the updated edge features, concatenates them each
               with the  node features of the *sending* node, and passes this tensor to
               an MLP. For each receiving node, the output of this MLP is then summed
               over in an aggregation step. These aggregated features are then
               concatenated with the features of the receiving node, and passed to
               another MLP. The output of this MLP then constitutes the updated node
               features.
            3. The global model is a simple global pooling of the node features, which
               are then passed to an MLP.

        For multiple "stacks" of graphs, steps 1 and 2 are repeated. Step 3 is only used
        for the final output of the graph.

        :param layers: number of `MetaLayer` graphs to construct
        :param hidden: latent space size of the edge, node, and global model MLPs; this
            also sets the size of the latent space representation of the edge and node
            features after a single MetaLayer pass
        :param lr: learning rate
        """
        super().__init__(lr=lr)
        self.layers = layers

        # list for multiple graph layers
        self.graphs = torch.nn.ModuleList()
        self.graphs.append(
            MetaLayer(
                EdgeModel(3, 1, hidden), NodeModel(3, hidden), GlobalModel(hidden, 1)
            )
        )

        # add multiple graph layers
        for _ in range(self.layers - 1):
            self.graphs.append(
                MetaLayer(
                    EdgeModel(hidden, hidden, hidden),
                    NodeModel(hidden, hidden),
                    GlobalModel(hidden, 1),
                )
            )

    def forward(self, x, edge_index, edge_attr, u, batch):
        for aa in range(self.layers):
            x, edge_attr, u = self.graphs[aa](x, edge_index, edge_attr, None, batch)

        return u


class InteractionSubSystem(BaseNet):
    """Class to build subgraphs based on Pnet biological subprocesses, and pass these
    subgraphs as input to an `InteractionNetworkMulti`.
    """

    def __init__(
        self,
        model_config,
        activation=None,
        node_subset=None,
        max_nodes=None,
        lr: float = 0.001,
    ):
        super().__init__(lr=lr)
        # Note: this assumes each graph has the same number of max_nodes
        # self.node_subset = [n+max_nodes*i for i in range(batch) for n in node_subset]
        self.node_subset = np.array(node_subset) if node_subset is not None else None
        self.max_nodes = max_nodes
        self.activation_fn = activation
        self.interactionnetwork = InteractionNetworkMulti(
            layers=model_config.get("layers"), hidden=model_config.get("hidden")
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if self.node_subset is not None:
            bs = int(batch.max()) + 1
            assert batch.shape[0] == bs * self.max_nodes
            batch_subset = np.concatenate(
                [self.node_subset + self.max_nodes * i for i in range(bs)], axis=0
            ).tolist()
            edge_index, edge_attr = subgraph(
                subset=batch_subset,
                edge_index=edge_index,
                edge_attr=edge_attr,
                relabel_nodes=True,
            )
            x = x[batch_subset]
            batch = batch[batch_subset]

        u = self.interactionnetwork(x, edge_index, edge_attr, None, batch)
        if self.activation_fn is not None:
            u = self.activation_fn(u)
        return u


class VisibleDense(BaseNet):
    def __init__(self, pathway_map, activation=None, use_bias=True, lr: float = 0.001):
        super().__init__(lr=lr)
        # import gene pathway map
        if isinstance(pathway_map, pd.DataFrame):
            self.map = pathway_map.to_numpy().astype(np.float32)
        else:
            self.map = pathway_map.astype(np.float32)

        self.units = self.map.shape[1]
        self.input_dim = self.map.shape[0]
        self.use_bias = use_bias
        # identify indices where we have connections
        self.nonzero_ind = torch.LongTensor(np.array(np.nonzero(self.map)).T)
        nonzero_count = self.nonzero_ind.shape[0]

        # build a tensor of parameters to hold the nonzero weights
        self.kernel_vector = torch.nn.Parameter(torch.zeros(nonzero_count))
        # potentially problematic line - distributes the nonzero indices to a matrix of
        # zeroes

        # builds a tensor to hold the biases
        self.bias = torch.nn.Parameter(
            data=torch.zeros(
                self.units,
            )
        )
        # initialise the weights and biases
        torch.nn.init.uniform_(self.kernel_vector, -0.01, 0.01)
        torch.nn.init.uniform_(self.bias, -0.01, 0.01)
        self.activation_fn = activation

    def forward(self, x):
        # XXX should get rid of calls to "to"
        device = self.kernel_vector.device
        self.kernel = scatter_nd(
            self.nonzero_ind, self.kernel_vector, shape=(self.input_dim, self.units)
        )
        # multiply input vector with the sparse matrix
        out = torch.matmul(x, self.kernel.to(device))
        if self.use_bias:
            out = out + self.bias
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out


class VisibleGraphInteractionNet(BaseNet):
    """Model that combines a stack of `MetaLayer`s composed of subgraphs with a final
    neural layer.
    """

    def __init__(
        self,
        pathway_maps,
        node_index,
        model_config=None,
        sparse=False,
        lr: float = 0.001,
    ):
        """Initialize the net.

        :param sparse: if `False`, uses a single fully connected layer after the graph
            outputs; if `True`, uses sparse connections for the final layer, based off
            Pnet masks
        :param lr: learning rate
        """
        super().__init__(lr=lr)
        self.model_config = model_config
        self.pathway_maps = pathway_maps
        self.node_index = node_index
        self.pathway_to_nodes = self.get_node_subset()
        self.subsys = torch.nn.ModuleList(
            [
                InteractionSubSystem(
                    model_config=self.model_config,
                    node_subset=self.pathway_to_nodes[target_pathway],
                    max_nodes=len(self.node_index),
                )
                for target_pathway in self.pathway_maps[0].columns
            ]
        )
        hidden = self.pathway_maps[1].shape[1]
        if sparse == False:
            self.nn = Sequential(
                Linear(len(self.subsys), hidden), ReLU(), Linear(hidden, 2)
            )
        else:
            self.nn = Sequential(
                VisibleDense(pathway_map=self.pathway_maps[1]),
                ReLU(),
                Linear(hidden, 2),
            )

    def get_node_subset(self):
        pathway_to_nodes = {}
        for target_pathway in self.pathway_maps[0].columns:
            subset = [
                self.pathway_maps[0].index[i]
                for i, g in enumerate(self.pathway_maps[0][target_pathway])
                if g == 1
            ]
            subset = sorted([self.node_index[g] for g in subset])
            pathway_to_nodes[target_pathway] = subset

        return pathway_to_nodes

    def forward(self, x, edge_index, edge_attr, batch):
        h = torch.cat([g(x, edge_index, edge_attr, batch) for g in self.subsys], dim=-1)
        out = self.nn(h)
        return F.log_softmax(out, dim=-1)
