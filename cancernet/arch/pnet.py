import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, ReLU

from cancernet.arch.base_net import BaseNet
from cancernet.util import scatter_nd


class FeatureLayer(torch.nn.Module):
    """This layer will take our input data of size `(N_genes, N_features)`, and perform
    elementwise multiplication of the features of each gene. This is effectively
    collapsing the `N_features dimension`, outputting a single scalar latent variable
    for each gene.
    """

    def __init__(self, num_genes: int, num_features: int, hidden: int=1):
        super().__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        self.hidden = hidden
        weights = torch.Tensor(self.num_genes, self.num_features, hidden)
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.Tensor(self.num_genes, hidden))
        # initialise weights using a normal distribution; can also try uniform
        torch.nn.init.normal_(self.weights, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x):
        x = torch.unsqueeze(x, -1) * self.weights
        x = torch.sum(x, dim=-2)
        x = x + self.bias
        x = x.squeeze(dim=-1)
        return x


class SparseLayer(torch.nn.Module):
    """Sparsely connected layer, with connections taken from pnet."""

    def __init__(self, layer_map):
        super().__init__()
        map_numpy = layer_map.to_numpy()
        self.register_buffer(
            "nonzero_indices", torch.LongTensor(np.array(np.nonzero(map_numpy)).T)
        )
        self.layer_map = layer_map
        self.shape = map_numpy.shape
        self.weights = nn.Parameter(torch.Tensor(self.nonzero_indices.shape[0]))
        self.bias = nn.Parameter(torch.Tensor(self.shape[1]))
        torch.nn.init.normal_(self.weights, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x):
        sparse_tensor = scatter_nd(self.nonzero_indices, self.weights, self.shape)
        x = torch.mm(x, sparse_tensor)
        # no bias yet
        x = x + self.bias
        return x


class PNet(BaseNet):
    """Implementation of the pnet sparse feedforward network in torch. Uses the same
    pytorch geometric dataset as the message passing networks.
    """

    def __init__(self, layers, num_genes: int, num_features: int, lr: float = 0.001):
        """Initialize.

        :param layers: list of pandas dataframes describing the pnet masks for each
            layer
        :param num_genes: number of genes in dataset
        :param num_features: number of features for each gene
        :param lr: learning rate
        """
        super().__init__(lr=lr)
        self.layers = layers
        self.num_genes = num_genes
        self.num_features = num_features
        self.network = nn.ModuleList()
        self.network.append(FeatureLayer(self.num_genes, self.num_features))
        self.network.append(nn.Tanh())
        self.network.append(nn.Dropout(p=0.5))
        for layer_map in layers:
            self.network.append(SparseLayer(layer_map))
            self.network.append(nn.Tanh())
            self.network.append(nn.Dropout(p=0.1))
        # final layer
        self.network.append(Linear(layer_map.to_numpy().shape[1], 2))

    def forward(self, data):
        """Only uses the "node features", which in this case we just treat as a data
        vector for the sparse feedforward network.
        """
        # reshape for batching appropriate for feedfoward network
        x = torch.reshape(
            data.x, (int(data.batch[-1] + 1), self.num_genes, self.num_features)
        )
        for hidden_layer in self.network:
            x = hidden_layer(x)
        return F.log_softmax(x, dim=-1)
