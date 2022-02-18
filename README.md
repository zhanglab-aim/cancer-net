# CancerGNN: A deep graph learning pipeline for genomic data analysis

## Installation

### Conda
Installation should work out of the box for Conda:

``` conda env create -f cancerenv1.yml -n cancerenv ```

Of course, you should first install Conda from [here](https://docs.conda.io/en/latest/miniconda.html).
Choose `yes` for running `conda init`. You can disable automatically activating the
base Conda environment by running `conda config --set auto_activate_base false`. This is
useful if you work with both `conda` and `venv` environments on the same machine.

### Venv
For now support is spotty for `venv` installation. You can start with

``` pip install -r requirements.txt ```

This sometimes fails for some packages, which can then be installed manually. Running
the `pip install -r requirements.txt` command again should then fix things.

If having trouble with `pyg`, try the instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


## Running

Training, testing and explaination are integrated in file `03-main.py`. To run

``` python 03-main.py ```

Use

``` python 03-main.py --help ```

for options.

## Folder structure -- this will change soon
```
.
├── data
│     └──raw                    # please put all the raw TCGA data here 
│     └──processed              # this will save the processed data via TCGAData.py
│
│
├──CancerGNN
    └──graph                    # intermediate files towards constructing the gene graph
    └──utils                    # a few util functions
    │    └──GNNExplainer.py     # explain the important subgraphs
    │    └──convert.py          # graph format transformation
    │    └──word_cloud.py       # visualize gene importance as word cloud
    └──arch                     # a few util functions
    │    └──net.py              # define the gnn models 
    └──01.py                    # convert gene ID encoding methods ( HumanBase and TCGA are not aligend)
    └──02.py                    # load human base graph into defualtdict piclke 
    └──03.py                    # main script that contains loading data, training, testing and explaination. it will call TCGAData.py, net.py
    └──TCGAData.py              # Define the pyg graph dataset class for TCGA data. it will call build_graph.py
    └──build_graph_ref.py       # build graph structure with minusing reference gene
```

## Outline

![pipeline](figures/cancer_gnn_pipeline.png?raw=true)

We treat the genomic data of each sample as a graph, where each gene stands for a node,
gene-gene interactions are defined as edges and taken from [HumanBase](https://hb.flatironinstitute.org/).
Each gene (node) is associated with node features. In this project,  we used pretained
neural networks to extract 128-dim features for each gene. Specifically, we used Xpresso
[1] to encode the non-coding regions; we used UniRep [2] to encode the coding regions. 

### Generating pre-processed gene graph -- this is subject to change

Run `01-network_id_conversion.py` to convert HumanBase to TCGA gene identifiers. Then
run `02-trasfer_hb_graph.py` to read the graph, symmetrize edges (the original data
contains edges in one direction only), and store as a Python pickle.

### References
[1] Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks, Agarwal et al. 2020, Cell Reports

[2] Unified rational protein engineering with sequence-only deep representation learning, Alley et al. 2019. Nature Methods

## Graph Neural Network (GNNs) for graph classification
We have implemented two kinds of GNNs: one is vanilla GNN (GCN [3]) and another one is a
more advanced GNN (GCNII[4]).

[3] Semi-supervised classification with graph convolutional networks, Kipf, Thomas N., and Max Welling. ICLR 2017

[4] Simple and Deep Graph Convolutional Networks, Chen et al. ICML 2020


## Other related work
[5] Deep generative models of genetic variation capture the effects of mutations, Riesselman, Adam J., et al. Nature Method 2018

[6] Classification of Cancer Types Using Graph Convolutional Neural Networks, Ramirez, Ricardo, et al. Frontiers in Physics 2020

[7] Integration of multiomics data with graph convolutional networks to identify new cancer genes and their associated molecular mechanisms, Roman Schulte-Sasse et al. Nature Machine Intelligence 2021

[8] Representation Learning for Networks in Biology and Medicine: Advancements, Challenges, and Opportunities, Michelle M. Li et al. arXiv preprint arXiv:2104.04883.



