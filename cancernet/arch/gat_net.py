import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import GCNConv, GlobalAttention

from typing import Sequence

from cancernet.arch.base_net import BaseNet
from typing import Tuple, List


class GATNet(BaseNet):
    def __init__(
        self,
        num_classes: int = 2,
        dims: Sequence = (128, 128, 64, 128),
        lr: float = 0.01,
    ):
        super().__init__(lr=lr)

        assert len(dims) == 4

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

        self.gate_nn = nn.Sequential(
            nn.Linear(dims[2], 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.pool = GlobalAttention(gate_nn=self.gate_nn)

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()

        # dimension stays 128
        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        # x = F.dropout(x, p=0.5, training=self.training)

        # dimension goes down to 64
        x1 = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        # x1 = F.dropout(x1, p=0.5, training=self.training)

        # global pooling leads us into non-graph neural net territory
        # x2 = global_mean_pool(x1, data.batch)
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
