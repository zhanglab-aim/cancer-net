"""Define functions for loading gene matrix transposed (`.gmt`) files."""

import re
import pandas as pd


def load_gmt(filename: str, genes_col: int = 1, pathway_col: int = 0) -> pd.DataFrame:
    """Load a `.gmt` file into a Pandas dataframe with columns `group` and `gene`.

    Each row of a `.gmt` file is a gene set.

    :param genes_col: first column representing genes; typically this is 1, but
        could be 2, e.g., if there is an information column between pathway and
        genes
    :param pathway_col: column for pathway names
    """
    data_dict_list = []
    with open(filename) as gmt:
        # reading manually because each row has different number of fields
        data_list = gmt.readlines()

        for row in data_list:
            elems = row.strip().split("\t")
            # XXX regex is overkill here
            elems = [re.sub("_copy.*", "", g) for g in elems]
            # XXX since elems are split from a stripped row, there can't be a
            #     newline there!
            elems = [re.sub("\\n.*", "", g) for g in elems]
            for gene in elems[genes_col:]:
                pathway = elems[pathway_col]
                dict = {"group": pathway, "gene": gene}
                data_dict_list.append(dict)

    df = pd.DataFrame(data_dict_list)

    return df
