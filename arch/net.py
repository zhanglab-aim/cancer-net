from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    NNConv,
    TopKPooling,
    GCNConv,
    GCNConv,
    GCN2Conv,
    PairNorm,
)
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention




def scatter_nd(indices: torch.Tensor, weights: torch.Tensor, shape) -> torch.Tensor:
    """ 
        For a list of indices and weights, return a sparse matrix of desired shape
        with weights only at the desired indices. Named after tensorflow.scatter_nd
    """
    
    ind1d = indices[:, 0]
    n = shape[0]
    for i in range(1, len(shape)):
        ind1d = ind1d * shape[i] + indices[:, i]
        n *= shape[i]
    ind1d.to(device) ## Indices of sparse weights
    weights.to(device) ## Vector of sparse weights 
    zz=torch.zeros(n,device=torch.device(device)) ## Initialise sparse matrix
    ## Now broadcast the weights across our sparse matrix
    sparse_matrix = zz.scatter_add_(0, ind1d, weights).reshape(*shape)
    return sparse_matrix

##############################################################################
#################################### Layers ##################################
##############################################################################

class FeatureLayer(torch.nn.Module):
    """
        This layer will take our input data of size [N_genes, N_features],
        and perform elementwise multiplication of the features of each gene.
        This is effectively collapsing the N_features dimension, outputting a
        single scalar latent variable for each gene.
    """
    def __init__(self,num_genes,num_features):
        super().__init__()
        self.num_genes=num_genes
        self.num_features=num_features
        weights=torch.Tensor(self.num_genes,self.num_features)
        self.weights=nn.Parameter(weights)
        self.bias=nn.Parameter(torch.Tensor(self.num_genes))
        ## Initialise weights using a normal distrubtion, can also try uniform
        torch.nn.init.normal_(self.weights,mean=0.0,std=1.0)
        torch.nn.init.normal_(self.bias,mean=0.0,std=1.0)
        
    def forward(self,x):
        x=x*self.weights
        x=torch.sum(x,dim=2)
        x=x+self.bias
        return x
    
class SparseLayer(torch.nn.Module):
    """
        Sparsely connected layer, with connections taken from pnet
    """
    def __init__(self,layer_map):
        super().__init__()
        map_numpy=layer_map.to_numpy()
        #self.nonzero_indices=torch.LongTensor(np.array(np.nonzero(map_numpy)).T)
        self.register_buffer("nonzero_indices",torch.LongTensor(np.array(np.nonzero(map_numpy)).T))
        self.shape=map_numpy.shape
        self.weights=nn.Parameter(torch.Tensor(self.nonzero_indices.shape[0]))
        torch.nn.init.normal_(self.weights,mean=0.0,std=1.0)
        
    def forward(self,x):
        sparse_tensor=scatter_nd(self.nonzero_indices,self.weights,self.shape)
        x=torch.mm(x,sparse_tensor)
        ## no bias yet
        return x


##############################################################################
############################## Graph Models ##################################
##############################################################################


class EdgeModel(torch.nn.Module):
    def __init__(self,node_size,edge_attr_size,hidden):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(node_size*2+edge_attr_size, hidden),
                            ReLU(),
                            Lin(hidden, hidden))

    def forward(self, src, dest, edge_attr, u, batch):
        """
         Function to update edge attribtes. Takes as input:
             - src: [E, F_x]  where E is the number of edges, F_x are the node features
                              of the sending node.
             - dest: [E, F_x] the node features for the "receiving" nodes.
             - edge_attr: [E, F_e] where F_e are the edge features.
             - u: global features, not currently used.
             - batch: [E] with max entry B - 1.

         returns: [E, F_h] where F_h is the size of the hidden layers. These constitute
                           the updated edge features after a "message pass" step.
        """
        if len(edge_attr.shape)==1:
            out = torch.cat([src, dest, edge_attr.reshape(-1, 1)], 1)
        else:
            out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self,input_size,hidden):
        super(NodeModel, self).__init__()
        self.message_function = Seq(Lin(input_size+hidden, hidden),
                              ReLU(),
                              Lin(hidden, hidden),
                              ReLU(),
                              Lin(hidden, hidden))
        self.node_mlp = Seq(Lin(input_size+hidden, hidden),
                              ReLU(),
                              Lin(hidden, hidden),
                              ReLU(),
                              Lin(hidden, hidden))

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
         Update node attributes - takes node features & edge features, updates node features
         based on the features of the sending and receiving node, and edge features of each connection.
             - x: [N, F_x] where N is the number of nodes, F_x are the node features. NB that
                           F_x can be different for different layers of the graph (i.e. the input
                           feature size is currently 3, while the latent node feature size is
                           of size `hidden`.
             - edge_index: [2,E] where E is the number of edges. List of indices describing the sending
                                 and receiving nodes of each edge.
             - edge_attr: [E, F_e] where F_e are the edge features. NB this can also be different
                                   for different layers.
             - u: Global features, not currently used.
             - batch: [E] with max entry B - 1.

         returns: [N, F_h] where F_h is the size of the hidden layers.
        """
        send_idx, rec_idx = edge_index ## Indices of sending and receiving nodes
        out = torch.cat([x[send_idx], edge_attr], dim=1) ## Tensor of node features of sending nodes, concatenated with the edge features
        out = self.message_function(out)
        out = scatter_add(out, rec_idx, dim=0, dim_size=x.size(0)) ## Aggregation step - for each receiving node, take the average of the hidden layer outputs connected to that node
        ## Finally concat each node feature with the hidden layer outputs, pass to one final MLP
        return self.node_mlp(torch.cat([x, out], dim=1))


class GlobalModel(torch.nn.Module):
    def __init__(self, hidden, outputs):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(hidden, hidden),
                              BatchNorm1d(hidden),
                              ReLU(),
                              Lin(hidden, outputs))

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
         Aggregate node features. Global mean pool, then pass the pooled features
         to a MLP.
        """
        out = scatter_mean(x, batch, dim=0)
        out = self.global_mlp(out)
        return out


class InteractionNetworkMulti(torch.nn.Module):
    def __init__(self,layers,hidden):
        """
         Class to stack multiple MetaLayers
            - layers: Number of MetaLayer graphs to construct
            - hidden: Sets the latent space size of the edge, node, and global model MLPs.
                      This also sets the size of the latent space representation of the edge
                      and node features after a single MetaLayer pass.
                      
        The general procedure for the MetaLayer is as follows.
            1. The EdgeModel takes the node features and edge features. For each edge connection,
               the node and edge features for each edge are concatenated together and passed to
               an MLP. The output is then a set of u pdated node features.
            2. The NodeModel takes the updated edge features, concatenates them each with the 
               node features of the *sending* node, and passes this tensor to an MLP.
               For each receiving node, the output of this MLP is then summed over in an aggregation step.
               These aggregated features are then concatenated with the features of the receiving node, and
               passed to another MLP. The output of this MLP then constitutes the updated node features
            3. The global model is a simple global pooling of the node features, which are then passed to an MLP
            
        For multiple "stacks" of graphs, steps 1 and 2 are repeated. Step 3 is only used for the final output of the graph.
        """
        super(InteractionNetworkMulti, self).__init__()
        self.layers=layers
        ## List for multiple graph layers
        self.graphs=nn.ModuleList()
        self.graphs.append(MetaLayer(EdgeModel(3,1,hidden), NodeModel(3,hidden), GlobalModel(hidden,1)))
        ## Add multiple graph layers
        for aa in range(self.layers-1):
            self.graphs.append(MetaLayer(EdgeModel(hidden,hidden,hidden), NodeModel(hidden,hidden), GlobalModel(hidden,1)))
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        x, edge_attr, u = self.graphs[0](x, edge_index, edge_attr, None, batch)
        for aa in range(1,self.layers):
            x, edge_attr, u = self.graphs[aa](x, edge_index, edge_attr, None, batch)
        return u

    
class InteractionSubSystem(torch.nn.Module):
    def __init__(self, model_config=None, activation=None, node_subset=None, max_nodes=None):
        """
         Class to build subgraphs based on Pnet biological subprocesses, and pass these subgraphs
         as input to an InteractionNetworkMulti
        """
        super(InteractionSubSystem, self).__init__()
        # Note: this assumes each graph has the same number of max_nodes
        #self.node_subset = [n+max_nodes*i for i in range(batch) for n in node_subset]
        self.node_subset =  np.array(node_subset)
        self.max_nodes = max_nodes
        self.activation_fn = activation
        self.interactionnetwork  = InteractionNetworkMulti(layers=model_config.get("layers"), hidden=model_config.get("hidden"))

    def forward(self, x, edge_index, edge_attr, batch):
        if self.node_subset is not None:
            bs = int(batch.max()) + 1
            assert batch.shape[0] == bs*self.max_nodes
            batch_subset = np.concatenate([self.node_subset+self.max_nodes*i for i in range(bs)], axis=0).tolist()
            edge_index, edge_attr = subgraph(subset=batch_subset, edge_index=edge_index, edge_attr=edge_attr, relabel_nodes=True)
            x = x[batch_subset]
            batch = batch[batch_subset]
        u = self.interactionnetwork(x, edge_index, edge_attr, None, batch)
        if self.activation_fn is not None:
            u = self.activation_fn(u)
        return u

##############################################################################
#################################### Models ##################################
##############################################################################

class pnet(torch.nn.Module):
    """ 
        Implementation of the pnet sparse feedforward network in torch
        Uses the same pytorch geometric dataset as the message passing networks
    """
        
    def __init__(self,layers,num_genes,num_features):
        """
            layers: A list of pandas dataframes describing the pnet masks
                   for each layer
            num_genes: number of genes in dataset
            num_features: number of features for each gene
        """
        super(pnet,self).__init__()
        self.layers=layers
        self.num_genes=num_genes
        self.num_features=num_features
        self.network = nn.ModuleList()
        self.network.append(FeatureLayer(self.num_genes,self.num_features))
        self.network.append(ReLU())
        for layer_map in layers:
            self.network.append(SparseLayer(layer_map))
            self.network.append(ReLU())
        ## Final layer
        self.network.append(Lin(layer_map.to_numpy().shape[1],2))
                            
    def forward(self, x, edge_index, edge_attr, batch):
        """
            Only uses the "node features", which in this case we just
            treat as a data vector for the sparse feedforward network
        """
        ## Reshape for batching appropriate for feedfoward network
        x=torch.reshape(x, (int(batch[-1]+1),self.num_genes, self.num_features))
        for hidden_layer in self.network:
            x=hidden_layer(x)
        return F.log_softmax(x, dim=-1)


class VisibleGraphInteractionNet(torch.nn.Module):
    def __init__(self, pathway_maps, node_index, model_config=None,sparse=False):
        """
         Model that combines a stack of MetaLayers composed of subgraphs with a final neural layer.
             - sparse: if False, uses a single fully connected layer after the graph outputs.
                       if True, uses sparse connections for the final layer, based off Pnet masks.
        """
        super(VisibleGraphInteractionNet, self).__init__()
        self.model_config = model_config
        self.pathway_maps = pathway_maps
        self.node_index = node_index
        self.pathway_to_nodes = self.get_node_subset()
        self.subsys = torch.nn.ModuleList([
                InteractionSubSystem(
                    model_config = self.model_config,
                    node_subset = self.pathway_to_nodes[target_pathway],
                    max_nodes = len(self.node_index)
                    )
                for target_pathway in self.pathway_maps[0].columns
                ])
        hidden = self.pathway_maps[1].shape[1]
        if sparse==False:
            self.nn = Seq(
                Lin(len(self.subsys), hidden),
                ReLU(),
                Lin(hidden,2)
            )
        else:
            self.nn = Seq(
                VisibleDense(pathway_map=self.pathway_maps[1]),
                ReLU(),
                Lin(hidden,2)
            )


    def get_node_subset(self):
        pathway_to_nodes = {}
        for target_pathway in self.pathway_maps[0].columns:
            subset = [self.pathway_maps[0].index[i] for i, g in enumerate(self.pathway_maps[0][target_pathway]) if g==1]
            subset = sorted([self.node_index[g] for g in subset])
            pathway_to_nodes[target_pathway] = subset
        return pathway_to_nodes

    def forward(self, x, edge_index, edge_attr, batch):
        h = torch.cat([g(x, edge_index, edge_attr, batch) for g in self.subsys], dim=-1)
        out = self.nn(h)
        return F.log_softmax(out, dim=-1)


class Net(torch.nn.Module):
    """A neural net based on edge-conditioned convolutions.
    
    This uses two edge-conditioned convolutions (`torch_geometric.nn.NNConv`), each
    followed by top-`k` pooling (`torch_geometric.nn.TopKPooling`), then uses global
    max and mean-pooling on the node attributes to generate features for an MLP that
    ultimately performs binary classification.

    :param dim: dimensionality of input node attributes
    """

    def __init__(self, dim: int = 128):
        super(Net, self).__init__()

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


class GATNet(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dims: Sequence = (128, 128, 64, 128),
        output_intermediate: bool = False,
    ):
        assert len(dims) == 4

        super(GATNet, self).__init__()

        # GCNConv basically averages over the node attributes of a node's neighbors,
        # weighting by edge weights (if given), and including the node itself in the
        # average (i.e., including a self-loop edge with weight 1). The average is also
        # weighted by the product of the square roots of the node degrees (including the
        # self-loops), and is finally transformed by a learnable linear layer.
        self.prop1 = GCNConv(in_channels=dims[0], out_channels=dims[1])
        self.prop2 = GCNConv(in_channels=dims[1], out_channels=dims[2])

        self.fc1 = torch.nn.Linear(dims[2], dims[3])
        self.fc2 = torch.nn.Linear(dims[3], num_classes)
        self.m = nn.LogSoftmax(dim=1)

        self.gate_nn = nn.Sequential(
            nn.Linear(dims[2], 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.pool = GlobalAttention(gate_nn=self.gate_nn)

        self.output_intermediate = output_intermediate

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
        if self.output_intermediate:
            return y, x1, x2
        else:
            return y


class GCNNet(torch.nn.Module):
    """A network based on graph convolutional operators.
    
    This applies a couple of graph convolutional operators followed by an MLP. Graph
    convolutional operators basically average over the node attributes from a given node
    plus its neighboring nodes, with weights proportional to edge weights and inversely
    proportional to the square roots of the node degrees. The final node attributes are
    obtained by passing through a fully-connected linear layer. See
    `torch_geometric.nn.GCNConv` for full details.

    The result from the graph convolutional layers is passed through an MLP, with class
    predictions obtained by (log) softmax.

    :param output_intermediate: if true, the module outputs not only the final class
        prediction, but also:
            `x1`: the result after the graph convolutional layers, passed through a ReLU
            `x2`: the result from the global mean pooling (before dropout)
    :param num_classes: number of output classes
    :param dims: dimensions of the input layers (`dims[0]`) and the various three hidden
        layers; should have length 4
    """

    def __init__(
        self,
        num_classes: int = 2,
        dims: Sequence = (128, 128, 64, 128),
        output_intermediate: bool = False,
    ):
        assert len(dims) == 4

        super(GCNNet, self).__init__()

        # GCNConv basically averages over the node attributes of a node's neighbors,
        # weighting by edge weights (if given), and including the node itself in the
        # average (i.e., including a self-loop edge with weight 1). The average is also
        # weighted by the product of the square roots of the node degrees (including the
        # self-loops), and is finally transformed by a learnable linear layer.
        self.prop1 = GCNConv(in_channels=dims[0], out_channels=dims[1])
        self.prop2 = GCNConv(in_channels=dims[1], out_channels=dims[2])

        self.fc1 = torch.nn.Linear(dims[2], dims[3])
        self.fc2 = torch.nn.Linear(dims[3], num_classes)
        self.m = nn.LogSoftmax(dim=1)

        self.output_intermediate = output_intermediate

    def forward(self, data):
        data.edge_attr = data.edge_attr.squeeze()

        # dimension stays 128
        x = F.relu(self.prop1(data.x, data.edge_index, data.edge_attr))
        # x = F.dropout(x, p=0.5, training=self.training)

        # dimension goes down to 64
        x1 = F.relu(self.prop2(x, data.edge_index, data.edge_attr))
        # x1 = F.dropout(x1, p=0.5, training=self.training)

        # global pooling leads us into non-graph neural net territory
        x2 = global_mean_pool(x1, data.batch)
        x = F.dropout(x2, p=0.5, training=self.training)

        # back to 128-dimensions, then down to the number of classes
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # (log) softmax for class predictions
        y = self.m(x)

        # if asked to, return some intermediate results
        if self.output_intermediate:
            return y, x1, x2
        else:
            return y


class GCN2Net(torch.nn.Module):
    """A much larger network based on extended graph convolutional operators.
    
    This uses a dropout followed by a fully-connected layer with ReLU to blow up the
    input node attributes to `hidden_channels`-dimensions, then passes the resulting
    graph through a series of graph convolutional operators with initial residual
    connections and identity mapping (GCNII). This basically averages attributes over
    neighboring nodes, like normal `GCN` (see `GCNNet`), but it includes "skip"
    connections to some "initial" representation, and it also "shrinks" the linear
    weights acting on top of the convolution result towards the identity, with stronger
    shrinkage for deeper layers. See `torch_geometric.nn.GCN2Conv` for full details.

    The convolutional operators are followed by pair normalization, which aims to avoid
    oversmoothing (see `torch_geometric.nn.PairNorm`). Dropout layers are used for
    regularization, one before each convolutional layer.

    The result from the graph convolutional layers is passed through an MLP, with class
    predictions obtained by (log) softmax. The MLP also includes a batchnorm layer.

    :param hidden_channels: dimensionality of the node attributes during the
        convolutional layers
    :param num_layers: number of `GCN2Conv` layers
    :param alpha: strength of initial connections in `GCN2Conv` layers
    :param theta: strength of identity mapping in `GCN2Conv` layers
    :param num_classes: number of output classes
    :param shared_weights: whether to use different weight matrices for the convolution
        result and the initial residual; see `torch_geometric.nn.GCN2Conv`
    :param dropout: dropout strength
    :param output_intermediate: if true, the module outputs not only the final class
        prediction, but also:
            `x1`: the result after the graph convolutional layers
            `x2`: the result from the global pooling
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        alpha: float,
        theta: float,
        dim: int = 128,
        num_classes: int = 2,
        shared_weights: bool = True,
        dropout: float = 0.0,
        output_intermediate: bool = False,
    ):
        super(GCN2Net, self).__init__()

        # ModuleList makes PyTorch aware of the parameters for each module in the list
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(dim, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels * 2, hidden_channels // 2))
        self.lins.append(torch.nn.Linear(hidden_channels // 2, num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            # GCN2Conv is a graph convolutional layer similar to GCNConv, where node
            # attributes are averaged over neighbors. There are two difference,
            # controlled by the parameters `alpha` and `theta`:
            #   * it has skip connections so that the result from the convolution is
            #     combined with an "initial feature representation" (which is passed to
            #     the `GCN2Conv` layer at call time), with weights `(1 - alpha)` and
            #     `alpha`
            #   * the fully-connected linear layer that is applied to each node
            #     attribute after the convolution operation is "shrunk" towards the
            #     identity by a layer-dependent amount `1 - beta`, which is related to
            #     the parameter `theta` below by `beta = log(theta / l + 1)`. Thus
            #     earlier layers exprience less shrinkage, while later layers are pulled
            #     close to the identity.
            # A `GCN2Conv` behaves just as a `GCNConv` when `alpha=0` and `theta=0`.
            self.convs.append(
                GCN2Conv(
                    hidden_channels,
                    alpha,
                    theta,
                    layer + 1,
                    shared_weights,
                    normalize=False,
                )
            )

        self.dropout = dropout

        # PairNorm is a normalization step meant to guard against excessive smoothing
        # from the graph convolutional layers. It centers each node attribute to the
        # mean across all nodes, and normalizes by the variance of all the node
        # attributes at all nodes (i.e., flattening over both nodes and components).
        self.pnorm = PairNorm()

        # batch normalization
        self.bn = nn.BatchNorm1d(hidden_channels // 2)
        self.m = nn.LogSoftmax(dim=1)
        self.output_intermediate = output_intermediate

    def forward(self, data):
        # XXX what is the point of doing ToSparseTensor transform in the pre-processing?
        #     GCN2Conv already adds self-loops, and how else is `adj_t` different from
        #     edge index?
        x, adj_t, batch = data.x, data.adj_t, data.batch
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)

            x = h + x

            x = x.relu()
            x = self.pnorm(x)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = global_sort_pool(x, batch, 1)

        # building some intermediate results and doing global pooling
        x1 = x
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # switch to usual MLP (with batchnorm) to get the final answer
        x = self.lins[1](x2)
        x = self.bn(x.relu())

        x = self.lins[2](x)

        # (log) softmax for class predictions
        y = self.m(x)

        # if asked to, return intermediate results
        if self.output_intermediate == True:
            return y, x1, x2
        else:
            return y
