import os
import pandas as pd

from cancernet.dataset import load_gmt


class Reactome:
    """Class to load reactome data.

    This consists of
    1. Mapping from pathway identifiers to pathway names (`pathway_names`).
    2. Sets of genes for each pathway name (`pathway_genes`), as a `.gmt` file.
    3. Set of `child`-`parent` relations between pathways (`relations` or `hierarchy`).
    """

    def __init__(
        self,
        reactome_base_dir: str,
        relations_file_name: str,
        pathway_names_file_name: str,
        pathway_genes_file_name: str,
    ):
        self.reactome_base_dir = reactome_base_dir
        self.relations_file_name = relations_file_name
        self.pathway_names_file_name = pathway_names_file_name
        self.pathway_genes_file_name = pathway_genes_file_name

        self.pathway_names = self.load_names()
        self.pathway_genes = self.load_genes()
        self.hierarchy = self.load_hierarchy()

    def load_names(self) -> pd.DataFrame:
        filename = os.path.join(self.reactome_base_dir, self.pathway_names_file_name)
        df = pd.read_csv(filename, sep="\t")
        df.columns = ["reactome_id", "pathway_name", "species"]
        return df

    def load_genes(self) -> pd.DataFrame:
        filename = os.path.join(self.reactome_base_dir, self.pathway_genes_file_name)
        df = load_gmt(filename, pathway_col=1, genes_col=3)
        return df

    def load_hierarchy(self) -> pd.DataFrame:
        filename = os.path.join(self.reactome_base_dir, self.relations_file_name)
        df = pd.read_csv(filename, sep="\t")
        df.columns = ["child", "parent"]
        return df
