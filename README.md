# CancerGNN: A deep graph learning pipeline for genomic data analysis

[![Python 3.9.7](https://img.shields.io/badge/python-3.9.7-yellow.svg)](https://www.python.org/downloads/release/python-397/)

## Installation

### Conda
Start by setting up the environment: switch to the repo folder and run

``` conda env create -f environment.yml ```

Of course, you should first install Conda from [here](https://docs.conda.io/en/latest/miniconda.html).
Choose `yes` for running `conda init`. You can disable automatically activating the
base Conda environment by running `conda config --set auto_activate_base false`. This is
useful if you work with both `conda` and `venv` environments on the same machine.

Next activate the environment and install the package using

``` 
conda activate cancerenv
pip install .
```

For development, make an editable install:

``` 
conda activate cancerenv
pip install -e .
```


## Data acquisition
NB the data files involved are ~20GB. If you want to store these outside the repo, we suggest setting `cancer-net/data` as a symlink to elsewhere on your system where you would like to store the data. Then proceed with the following steps:
1. Run `bash pull_data.sh` to download data files.
2. Preprocess the HumanBase graph connections by running `python3 01-network_id_conversion.py` inside the conda environment. This converts the HumanBase genes to TCGA gene identifiers.


## Running

Find notebooks in the [demos](demos/) folder.


### References
[1] Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks, Agarwal et al. 2020, Cell Reports

[2] Unified rational protein engineering with sequence-only deep representation learning, Alley et al. 2019. Nature Methods

## Graph Neural Network (GNNs) for graph classification (outdated)
We have implemented two kinds of GNNs: one is vanilla GNN (GCN [3]) and another one is a
more advanced GNN (GCNII[4]).

[3] Semi-supervised classification with graph convolutional networks, Kipf, Thomas N., and Max Welling. ICLR 2017

[4] Simple and Deep Graph Convolutional Networks, Chen et al. ICML 2020


## Other related work
[5] Deep generative models of genetic variation capture the effects of mutations, Riesselman, Adam J., et al. Nature Method 2018

[6] Classification of Cancer Types Using Graph Convolutional Neural Networks, Ramirez, Ricardo, et al. Frontiers in Physics 2020

[7] Integration of multiomics data with graph convolutional networks to identify new cancer genes and their associated molecular mechanisms, Roman Schulte-Sasse et al. Nature Machine Intelligence 2021

[8] Representation Learning for Networks in Biology and Medicine: Advancements, Challenges, and Opportunities, Michelle M. Li et al. arXiv preprint arXiv:2104.04883.



