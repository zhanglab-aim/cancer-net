import mygene
import gzip
import sys

network_fp = "data/prostate/prostate_gland.gz"
out_fp = "data/prostate/prostate_gland.geneSymbol.gz"
print(f"converting {network_fp} to {out_fp}")

entrez_network = {}
with gzip.GzipFile(network_fp, "r") as f:
    for line in f:
        try:
            g1, g2, score = line.strip().decode("utf-8").split()
        except ValueError:
            g1, g2, _, score = line.strip().decode("utf-8").split()
        entrez_network[(g1, g2)] = score

entrez_genes = set([g for k in entrez_network for g in k])
print("vertices=%i, edges=%i" % (len(entrez_genes), len(entrez_network)))

entrez_genes = list(entrez_genes)
mg = mygene.MyGeneInfo()
mapper = mg.querymany(
    entrez_genes, scopes="entrezgene", fields="symbol,ensemblgene", species="human"
)

entrez_to_symbol = {d["query"]: d["symbol"] for d in mapper if "symbol" in d}

symbol_network = {
    (entrez_to_symbol[gs[0]], entrez_to_symbol[gs[1]]): entrez_network[gs]
    for gs in entrez_network
    if gs[0] in entrez_to_symbol and gs[1] in entrez_to_symbol
}

print(
    "before conversion: %i\nafterconversion: %i"
    % (len(entrez_network), len(symbol_network))
)

with gzip.GzipFile(out_fp, "w") as fo:
    for g1, g2 in symbol_network:
        fo.write(("%s\t%s\t%s\n" % (g1, g2, symbol_network[(g1, g2)])).encode("utf-8"))

