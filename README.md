# CancerGNN: A deep graph learning pipeline for genomic data analysis

Codebase associated with [Reusability report: Prostate cancer stratification with diverse biologically-informed neural architectures](https://arxiv.org/abs/2309.16645). We re-implement the neural network architecture from [Biologically informed deep neural network for prostate cancer discovery](https://www.nature.com/articles/s41586-021-03922-4) in [PyTorch](https://pytorch.org/).

Additionally, we implement 3 different kinds of graph architectures, including a simple (graph convolutional network)[https://arxiv.org/abs/2007.02133], a (graph attention network)[https://arxiv.org/abs/1710.10903] and (MetaLayer)[https://arxiv.org/abs/1806.01261]. Graphs are constructured using gene connectivity patterns from the (HumanBase)[https://hb.flatironinstitute.org/] database.


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
3. The first time you create a `PnetDataSet`, the code will use these TCGA identifiers to construct a graph with connections between genes meeting a certain connection threshold (we have set `0.5` in our results). This can take ~20 minutes to construct, and so this graph is cached and saved as a pickle object. Next time a `PnetDataSet` is intialised, it will load the cached graph if it can find one, instead of reconstructing the graph every time.


## Running

Find example notebooks in the [demos](demos/) folder.

