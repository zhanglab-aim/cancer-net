# CancerGNN: A deep graph learning pipeline for genomic data analysis

Codebase associated with [Reusability report: Prostate cancer stratification with diverse biologically-informed neural architectures](https://arxiv.org/abs/2309.16645). We re-implement the neural network architecture from [Biologically informed deep neural network for prostate cancer discovery](https://www.nature.com/articles/s41586-021-03922-4) in [PyTorch](https://pytorch.org/).

Additionally, we implement 3 different kinds of graph architectures, including a simple [graph convolutional network](https://arxiv.org/abs/2007.02133), a [graph attention networ]([https://arxiv.org/abs/1710.10903) and [MetaLayer](https://arxiv.org/abs/1806.01261). Graphs are constructured using gene connectivity patterns from the [HumanBase](https://hb.flatironinstitute.org/) database.


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
2. Preprocess the HumanBase graph connections by running `python3 01-network_id_conversion.py` inside the conda environment. This converts the HumanBase genes in entrez ID to HGNC gene symbol identifiers in TCGA. 
3. The first time you create a `PnetDataSet`, the code will use these HGNC gene symbol identifiers to construct a graph with connections between genes meeting a certain connection threshold (we have set `0.5` in our results). This can take ~20 minutes to construct, and so this graph is cached and saved as a pickle object. **NB this process requires a high memory node, ~128GB**. Next time a `PnetDataSet` is intialised, it will load the cached graph if it can find one, instead of reconstructing the graph every time.


## Running

Example notebooks in [demos](demos/) folder. These notebooks will load a dataset, split into train/validation/test splits, and train a neural network on the data. These notebooks also include calculations of various performance metrics.

## Reusability report
To produce the results in [Reusability report: Prostate cancer stratification with diverse biologically-informed neural architectures](https://arxiv.org/abs/2309.16645), we used the scripts in [reprod_report](reprod_report/). This folder contains scripts for both the hyperparameter sweeps and initialisation variance tests. We use [Weights and biases](https://wandb.ai/) to monitor model training and performance, so these scripts are reliant on `wandb`. Additionally, model weights and performance metrics are saved locally in the `wandb` save directory. NB that `wandb` results for the runs in [abs/2309.16645](https://arxiv.org/abs/2309.16645) can be viewed at the following links:
1. [GCN Sweep](https://wandb.ai/cancer-net/hyperparam_sweeps_May/sweeps/u6dqzdbo)
2. [GCN Variance](https://wandb.ai/cancer-net/init_variance_June/sweeps/wah6adqh)
3. [GAT Sweep](https://wandb.ai/cancer-net/hyperparam_sweeps_May/sweeps/6zwbz942)
4. [GAT Variance](https://wandb.ai/cancer-net/init_variance_June/sweeps/2nfi7med)
5. [MetaLayer Sweep](https://wandb.ai/cancer-net/hyperparam_sweeps_May/sweeps/i6236e2e)
6. [MetaLayer Variance](https://wandb.ai/cancer-net/init_variance_June/sweeps/vvbn5aco)
