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
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x * self.weights
        x = torch.sum(x, dim=-1)
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
        self.weights = nn.Parameter(torch.Tensor(self.nonzero_indices.shape[0],1))
        self.bias = nn.Parameter(torch.Tensor(self.shape[1]))
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        sparse_tensor = scatter_nd(self.nonzero_indices, self.weights.squeeze(), self.shape)
        x = torch.mm(x, sparse_tensor)
        # no bias yet
        x = x + self.bias
        return x

    
class PNet(BaseNet):
    """Implementation of the pnet sparse feedforward network in torch. Uses the same
    pytorch geometric dataset as the message passing networks.
    """

    def __init__(self, layers, num_genes: int, num_features: int=3, lr: float = 0.001, intermediate_outputs: bool=True):
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
        self.intermediate_outputs = intermediate_outputs
        self.network = nn.ModuleList()
        self.intermediate_outs = nn.ModuleList()
        self.network.append(nn.Sequential(FeatureLayer(self.num_genes, self.num_features),nn.Tanh()))
        self.loss_weights=[2, 7, 20, 54, 148, 400] ## Taken from pnet - final output layer is the last element
        if len(self.layers) > 5:
            self.loss_weights = [2]*(len(self.layers)-5) + self.loss_weights
        for i, layer_map in enumerate(layers):
            if i != (len(layers) - 1):
                if i==0:
                    ## First layer has dropout of 0.5, the rest have 0.1
                    dropout=0.5
                else:
                    dropout=0.1
                    ## Build pnet layers
                self.network.append(
                    nn.Sequential(nn.Dropout(p=dropout),SparseLayer(layer_map),nn.Tanh())
                                    )
                    ## Build layers for intermediate output
                if self.intermediate_outputs:
                    self.intermediate_outs.append(
                    nn.Sequential(nn.Linear(layer_map.shape[0], 1),nn.Sigmoid())
                                                  )
            else:
                self.network.append(nn.Sequential(nn.Linear(layer_map.shape[0], 1),nn.Sigmoid()))
                
    def forward(self, data):
        """Only uses the "node features", which in this case we just treat as a data
        vector for the sparse feedforward network.
        """
        ## Create list for each component of the loss:
        # reshape for batching appropriate for feedfoward network
        x=torch.reshape(
            data.x, (int(data.batch[-1] + 1), self.num_genes, self.num_features)
        )
        y = []
        
        ## update hidden states x while record the intermediate 
        ## layer outcome predictions y
        x=self.network[0](x)
        for aa in range(1,len(self.network)-1):
            y.append(self.intermediate_outs[aa-1](x))
            x=self.network[aa](x)
        y.append(self.network[-1](x))
        
        return y
    
    def step(self, batch, kind: str) -> dict:
        """ Step function executed by lightning trainer module.
        """
        # run the model and calculate loss
        y_hat = self(batch)

        if kind=="train":
            loss=0
            for aa,y in enumerate(y_hat):
                ## Here we take a weighted average of the preditive outputs. Intermediate layers first
                loss+=(self.loss_weights[aa]*self.criterion(y.squeeze(), batch.y.to(torch.float32)))
            loss/=np.sum(self.loss_weights[aa])
        else:
            loss=self.criterion(y_hat[-1].squeeze(),batch.y.to(torch.float32))

        correct = ((y_hat[-1]>0.5).flatten()==batch.y).sum()
        # assess accuracy
        total = len(batch.y)
        batch_dict = {
            "loss": loss,
            # correct and total will be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dict
