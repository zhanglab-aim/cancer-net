import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import GCNConv, GlobalAttention

from typing import Sequence
import numpy as np
from cancernet.arch.base_net import BaseNet
from cancernet.arch.pnet import FeatureLayer, SparseLayer
from cancernet.util.tensor import scatter_nd
from typing import Tuple, List


class SparsePool(torch.nn.Module):
    def __init__(self, layer_map):
        super().__init__()
        map_numpy = layer_map.to_numpy()
        self.layer_map = map_numpy
        self.register_buffer(
            "nonzero_indices", torch.LongTensor(np.array(np.nonzero(map_numpy)).T)
        )
        self.shape = map_numpy.shape
        weights = scatter_nd(
            self.nonzero_indices, torch.ones(self.nonzero_indices.shape[0]), self.shape
        )
        weights /= torch.Tensor([map_numpy.sum(axis=0).tolist()] * self.shape[0])
        self.register_buffer("weights", weights)

    def forward(self, x):
        # first, average the node feature dim from (bs, n_nodes, n_feats) to (bs, n_nodes)
        x = torch.mean(x, dim=-1)
        # then, average the nodes to pathways
        x = torch.mm(x, self.weights)
        return x


class VgnNet(BaseNet):
    def __init__(
        self,
        layers,
        dims: Sequence,
        lr: float = 0.01,
    ):
        super().__init__(lr=lr)

        # GCNConv basically averages over the node attributes of a node's neighbors,
        # weighting by edge weights (if given), and including the node itself in the
        # average (i.e., including a self-loop edge with weight 1). The average is also
        # weighted by the product of the square roots of the node degrees (including the
        # self-loops), and is finally transformed by a learnable linear layer.
        self.prop1 = GCNConv(in_channels=dims[0], out_channels=dims[1])
        self.prop2 = GCNConv(in_channels=dims[1], out_channels=dims[2])

        self.layers = layers
        assert len(self.layers) == 2, NotImplementedError("Only support 2-layer map")
        self.num_nodes = self.layers[0].shape[0]
        components = [
            # FeatureLayer(self.num_nodes, dims[2]),
            # nn.ReLU(),
            # nn.Dropout(p=0.2)
            SparseLayer(layer_map=self.layers[0]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            SparseLayer(layer_map=self.layers[1]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            # final layer
            nn.Linear(self.layers[1].shape[1], 2),
        ]
        self.pnet_layers = nn.Sequential(*components)

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()

        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)

        x = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)

        bs = data.batch[-1] + 1
        x = torch.reshape(x, (bs, self.num_nodes, -1))
        x = torch.mean(x, dim=-1)
        x = nn.Dropout(p=0.1)(x)
        x = self.pnet_layers(x)
        y = F.log_softmax(x, dim=-1)
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
