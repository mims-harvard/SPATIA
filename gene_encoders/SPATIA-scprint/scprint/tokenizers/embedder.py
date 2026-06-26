import os

import pandas as pd
import torch
from scprint import utils
from scprint.tokenizers.protein_embedder import PROTBERT

from torch.nn import AdaptiveAvgPool1d


def protein_embeddings_generator(
    genedf: pd.DataFrame,
    organism: str = "homo_sapiens",
    cache: bool = True,
    fasta_path: str = "/tmp/data/fasta/",
    embedding_size: int = 512,
):
    utils.load_fasta_species(species=organism, output_path=fasta_path, cache=cache)
    fasta_file = next(
        file for file in os.listdir(fasta_path) if file.endswith(".all.fa.gz")
    )
    protgenedf = genedf[genedf["biotype"] == "protein_coding"]
    utils.utils.run_command(["gunzip", fasta_path + fasta_file])
    utils.subset_fasta(
        protgenedf.index.tolist(),
        subfasta_path=fasta_path + "subset.fa",
        fasta_path=fasta_path + fasta_file[:-3],
        drop_unknown_seq=True,
    )
    prot_embedder = PROTBERT()
    prot_embeddings = prot_embedder(
        fasta_path + "subset.fa", output_folder=fasta_path + "esm_out/", cache=cache
    )
    utils.utils.run_command(["gzip", fasta_path + fasta_file[:-3]])
    m = AdaptiveAvgPool1d(embedding_size)
    prot_embeddings = pd.DataFrame(
        data=m(torch.tensor(prot_embeddings.values)), index=prot_embeddings.index
    )
    return prot_embeddings
