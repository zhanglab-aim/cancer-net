import networkx as nx
import re

import itertools
import numpy as np
import pandas as pd
import logging

from typing import List, Dict, Sequence

from cancernet.dataset import Reactome


class ReactomeNetwork:
    def __init__(self, reactome_kws):
        # low level access to reactome pathways and genes
        self.reactome = Reactome(**reactome_kws)
        # build a graph based on the reactome
        self.netx = self.get_reactome_networkx()

    def get_terminals(self):
        """Find the terminal nodes (leaves) of the network."""
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes

    def get_roots(self):
        """Find the children of the root node -- these are the roots of the independent
        graphs loaded by `get_reactome_networkx`.
        """
        # XXX there are more efficient ways of finding the children of a node...
        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    def get_reactome_networkx(self) -> nx.Graph:
        """Build a directed graph (`nx.DiGraph`) representation of the reactome.

        The reactome is made up of several connected components. One main `root` node is
        added as a parent to all of these to generate one connected graph.
        """
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy["child"].str.contains("HSA")]

        # build nx.DiGraph
        net = nx.from_pandas_edgelist(
            human_hierarchy, "child", "parent", create_using=nx.DiGraph()
        )
        net.name = "reactome"

        # add root node to connect all the connected components
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = "root"
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    # NB: nx.info is deprecated
    # def info(self):
    #     return nx.info(self.netx)

    def get_tree(self):
        """Find a spanning tree for the network."""
        # convert to tree
        G = nx.bfs_tree(self.netx, "root")

        return G

    def get_completed_network(self, n_levels: int) -> nx.Graph:
        """XXX What exactly does this do?"""
        G = complete_network(self.netx, n_levels=n_levels)
        return G

    def get_completed_tree(self, n_levels: int) -> nx.Graph:
        """XXX What exactly does this do?"""
        G = self.get_tree()
        G = complete_network(G, n_levels=n_levels)
        return G

    def get_layers(
        self, n_levels: int, direction: str = "root_to_leaf"
    ) -> List[Dict[str, List[str]]]:
        """XXX What exactly does this do?"""
        if direction == "root_to_leaf":
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels : 5]

        # get the last layer (genes level)
        terminal_nodes = [
            n for n, d in net.out_degree() if d == 0
        ]  # set of terminal pathways
        # we need to find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub("_copy.*", "", p)
            genes = genes_df[genes_df["group"] == pathway_name]["gene"].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes

        layers.append(dict)
        return layers


# XXX what are these functions for?


def get_layer_maps(
    reactome: ReactomeNetwork,
    genes,
    n_levels: int,
    direction: str,
    add_unk_genes: bool = False,
    verbose: bool = False,
) -> List[pd.DataFrame]:
    """XXX What does this do exactly?"""
    reactome_layers = reactome.get_layers(n_levels, direction)
    # add sort for reproducibility; FZZ 2022.10.12
    filtering_index = sorted(genes)
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        if verbose:
            # XXX it's actually layer n - i - 1, since we flipped reactome_layers!
            print("layer #", i)
        crt_map = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        if verbose:
            print("filtered_map", filter_df.shape)
        filtered_map = filter_df.merge(
            crt_map, right_index=True, left_index=True, how="left"
        )
        if verbose:
            print("filtered_map", filter_df.shape)

        if add_unk_genes:
            # add a node for genes without known reactome annotation
            if verbose:
                print("UNK")
            filtered_map["UNK"] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, "UNK"] = 1

        filtered_map = filtered_map.fillna(0)
        if verbose:
            print("filtered_map", filter_df.shape)
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        if verbose:
            # XXX it's actually layer n - i - 1, since we flipped reactome_layers!
            logging.info(
                "layer {} , # of edges  {}".format(i, filtered_map.sum().sum())
            )
        # sort rows and cols for reproducibility; FZZ 2020.10.12
        filtered_map = filtered_map[sorted(filtered_map.columns)]
        filtered_map = filtered_map.loc[sorted(filtered_map.index)]
        maps.append(filtered_map)
    return maps


def get_map_from_layer(layer_dict: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """XXX What does this do exactly?"""
    pathways = list(layer_dict.keys())
    print("pathways", len(pathways))
    genes = list(itertools.chain.from_iterable(list(layer_dict.values())))
    # add sort for reproducibility; FZZ 2022.10.12
    genes = sorted(list(np.unique(genes)))
    print("genes", len(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in list(layer_dict.items()):
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    return df.T


# utility functions used by the ReactomeNetwork


def add_edges(G: nx.Graph, node: str, n_levels: int) -> nx.Graph:
    """Make a chain from `node` to `n_levels` copies, building the copies if needed.

    node -> node_copy1 -> ... -> node_copy<n_levels>
    """
    edges = []
    source = node
    for l in range(n_levels):
        target = node + "_copy" + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def complete_network(G: nx.Graph, n_levels: int = 4) -> nx.Graph:
    """XXX What exactly does this do?"""
    # subgraph of neighbors within given radius from root
    sub_graph = nx.ego_graph(G, "root", radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source="root", target=node))
        # XXX doesn't choosing the ego_graph ensure that distance <= n_levels?
        if distance <= n_levels:
            diff = n_levels - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph


def get_nodes_at_level(net: nx.Graph, distance: int) -> List[str]:
    """Get all the nodes within the given `distance` from the `root`."""
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, "root", radius=distance))

    # XXX a single BFS traversal of the graph should be enough
    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1:
        nodes -= set(nx.ego_graph(net, "root", radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net: nx.Graph, n_levels: int) -> List[Dict[str, List[str]]]:
    """XXX What exactly does this do?"""
    layers = []
    for i in range(n_levels):
        # XXX a single traversal should be enough to find all the distances
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            # find "original" node that copies came from
            # XXX using a regex for this is terribly inefficient
            n_name = re.sub("_copy.*", "", n)
            next = net.successors(n)
            dict[n_name] = [re.sub("_copy.*", "", nex) for nex in next]
        layers.append(dict)

    return layers
