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

    def __init__(self, num_genes: int, num_features: int):
        super().__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        weights = torch.Tensor(self.num_genes, self.num_features)
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.Tensor(self.num_genes,))
        # initialise weights using a normal distribution; can also try uniform
        torch.nn.init.normal_(self.weights, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x):
        x = x * self.weights
        x = torch.sum(x, dim=-1)
        x = x + self.bias
        return x

class DiagonalLayer(torch.nn.Module):
    """This layer will take our input data of size `(N_genes, N_features)`, and perform
    elementwise multiplication of the features of each gene. This is effectively
    collapsing the `N_features dimension`, outputting a single scalar latent variable
    for each gene.
    """

    def __init__(self, num_genes: int, num_features: int):
        super().__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        input_dimension = self.num_genes * self.num_features
        self.shape = (input_dimension, self.num_genes)
        rows = np.arange(input_dimension)
        cols = np.arange(self.num_genes)
        cols = np.repeat(cols, self.num_features)
        nonzero_ind = np.column_stack((rows, cols))
        self.register_buffer(
            "nonzero_indices", torch.LongTensor(nonzero_ind)
        )
        weights = torch.Tensor(self.num_genes*self.num_features)
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.Tensor(self.num_genes))
        # initialise weights using a normal distribution; can also try uniform
        torch.nn.init.normal_(self.weights, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x):
        sparse_tensor = scatter_nd(self.nonzero_indices, self.weights, self.shape)
        x = torch.mm(x, sparse_tensor)
        # no bias yet
        x = x + self.bias
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
        self.criterion = nn.BCELoss()
        self.layers = layers
        self.num_genes = num_genes
        self.num_features = num_features
        self.network = nn.ModuleList()
        self.network.append(FeatureLayer(self.num_genes, self.num_features))
        self.network.append(nn.Tanh())
        self.network.append(nn.Dropout(p=0.5))
        for i, layer_map in enumerate(layers):
            if i != (len(layers) - 1):
                self.network.append(SparseLayer(layer_map))
                self.network.append(nn.Tanh())
                self.network.append(nn.Dropout(p=0.1))
            else:
                self.network.append(nn.Linear(layer_map.shape[0], 1))
                self.network.append(nn.Sigmoid())
        # final layer
        # this is not needed because reactome always connects to one root node
        # self.network.append(Linear(layer_map.to_numpy().shape[1], 1))

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
        return x
    
    def step(self, batch, kind: str) -> dict:
        """Generic step function that runs the network on a batch and outputs loss
        and accuracy information that will be aggregated at epoch end.

        This function is used to implement the training, validation, and test steps.
        """
        # run the model and calculate loss
        y_hat = self(batch).squeeze()

        loss = self.criterion(y_hat, batch.y.to(torch.float32))

        # assess accuracy
        pred = y_hat>0.5
        correct = pred.eq(batch.y).sum().item()

        total = len(batch)

        batch_dict = {
            "loss": loss,
            # correct and total will be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dict
