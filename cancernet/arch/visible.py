import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from cancernet.arch import BaseNet
from cancernet.arch import InteractionSubSystem, VisibleDense


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
