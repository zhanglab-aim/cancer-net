"""A metalayer graph net."""

import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch.nn.functional as F
from torch_scatter import scatter_mean

from torch_geometric.nn import MetaLayer

from cancernet.arch.base_net import BaseNet
from typing import Dict, Any


inputs = 3
#hidden = 128
outputs = 2


class EdgeModel(torch.nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(inputs * 2, hidden),
            BatchNorm1d(hidden),
            ReLU(),
            Linear(hidden, hidden),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.node_mlp_1 = Sequential(
            Linear(inputs + hidden, hidden),
            BatchNorm1d(hidden),
            ReLU(),
            Linear(hidden, hidden),
        )
        self.node_mlp_2 = Sequential(
            Linear(inputs + hidden, hidden),
            BatchNorm1d(hidden),
            ReLU(),
            Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.global_mlp = Sequential(
            Linear(hidden, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, outputs)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_mean(x, batch, dim=0)
        return self.global_mlp(out)


class InteractionNet(BaseNet):
    def __init__(self, lr: float = 0.01, hidden: float = 128):
        super().__init__(lr=lr)
        self.meta_layer = MetaLayer(EdgeModel(hidden), NodeModel(hidden), GlobalModel(hidden))

    def forward(self, data):
        x, edge_attr, u = self.meta_layer(
            data.x, data.edge_index, data.edge_attr, None, data.batch
        )
        return F.log_softmax(u, dim=-1)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Set up optimizers and schedulers.

        This adds a simple scheduler.
        """
        optimizer_list, _ = super().configure_optimizers()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_list[0], "min", factor=0.5, patience=20
        )

        return {
            "optimizer": optimizer_list[0],
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
            "frequency": 1,
        }
