import torch
import numpy as np
import pandas as pd

from cancernet.arch import BaseNet
from cancernet.util import scatter_nd


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
