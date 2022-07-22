from distutils.core import setup

setup(
    name="cancernet",
    version="0.0.5",
    url="https://github.com/ttesileanu/cancer-net",
    packages=["cancernet"],
    install_requires=[
        "setuptools",
        "black",
        "cudatoolkit>=11.3",
        "torch>=1.11",
        "torchvision",
        "torchaudio",
        "pytorch-lightning>=1.6",
        "pyg",
        "pysr",
        "mygene",
        "numpy",
        "scipy",
        "tqdm",
        "pandas",
        "scikit-learn",
        "h5py",
        "wandb",
        "optuna",
        "matplotlib",
        "seaborn",
        "jupyter",
    ],
)
