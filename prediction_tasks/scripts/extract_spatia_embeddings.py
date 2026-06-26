
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_PLACEHOLDER_METADATA = {
    "organism_ontology_term_id": "NCBITaxon:9606",
    "assay_ontology_term_id": "EFO:0022615",
    "self_reported_ethnicity_ontology_term_id": "unknown",
    "sex_ontology_term_id": "unknown",
    "cell_type_ontology_term_id": "unknown",
    "tissue_ontology_term_id": "UBERON:0002107",
    "disease_ontology_term_id": "PATO:0000461",
    "development_stage_ontology_term_id": "HsapDv:0000266",
    "donor_id": "unknown",
}


def fill_missing_metadata(adata: sc.AnnData, dataset_name: str) -> sc.AnnData:
    for col, val in _PLACEHOLDER_METADATA.items():
        if col not in adata.obs.columns:
            adata.obs[col] = val
    if "donor_id" not in adata.obs.columns or (adata.obs["donor_id"] == "unknown").all():
        adata.obs["donor_id"] = dataset_name
    if "dataset_name" not in adata.obs.columns:
        adata.obs["dataset_name"] = dataset_name
    if "index" not in adata.obs.columns:
        adata.obs["index"] = adata.obs.index.astype(str)
    return adata


def extract_embeddings(
    checkpoint: str,
    adata_path: str,
    output_path: str,
    lmdb_dir: str | None = None,
    max_len: int = 1000,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda",
    seed: int = 0,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("medium")

    from scdataloader import Preprocessor
    from scprint.tasks.cell_emb_spatial import Embedder
    from scprint.model.model_spatial import scPrint

    log.info(f"Loading checkpoint: {checkpoint}")
    model = scPrint.load_from_checkpoint(checkpoint, precpt_gene_emb=None)
    model.embs = None
    log.info(f"Model loaded: {type(model).__name__}, d_model={model.d_model}")

    log.info(f"Loading dataset: {adata_path}")
    adata = sc.read(adata_path)
    if adata.raw is not None:
        adata = adata.raw.to_adata()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    n_cells_original = adata.n_obs
    log.info(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    dataset_name = Path(adata_path).stem
    adata = fill_missing_metadata(adata, dataset_name)

    preprocessor = Preprocessor(
        do_postp=False,
        force_preprocess=True,
        min_nnz_genes=None,
        min_valid_genes_id=30,
    )
    adata = preprocessor(adata)
    if adata.n_obs < n_cells_original:
        log.warning(
            f"Preprocessor filtered {n_cells_original - adata.n_obs} cells "
            f"({adata.n_obs} remain)"
        )

    embedder = Embedder(
        batch_size=batch_size,
        num_workers=num_workers,
        how="random expr",
        max_len=max_len,
        add_zero_genes=0,
        doclass=False,
        output_expression="none",
        pred_embedding=["cell_type_ontology_term_id"],
        keep_all_cls_pred=False,
        fix_missing_image=(lmdb_dir is not None),
    )

    embed_kwargs: dict = {
        "model": model,
        "adata": adata.copy(),
        "cache": False,
    }
    if lmdb_dir is not None:
        log.info(f"Spatial mode: using LMDB images from {lmdb_dir}")
        embed_kwargs["spatial_datadir"] = lmdb_dir
        embed_kwargs["image_preprocesser"] = model.clip_model_type
    else:
        log.info("Gene-only mode (no LMDB images)")

    log.info("Extracting embeddings ...")
    result_adata, _ = embedder(**embed_kwargs)

    embedding: np.ndarray = result_adata.obsm["scprint"]
    log.info(f"Embedding shape: {embedding.shape}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embedding)
    log.info(f"Saved to {out}")

    return embedding


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SPATIA-scprint cell embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to scPRINT Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--adata_path", required=True,
        help="Path to input h5ad file",
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Output path for embeddings (.npy)",
    )
    parser.add_argument(
        "--lmdb_dir", default=None,
        help="Path to LMDB image database (spatial mode). Omit for gene-only mode.",
    )
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    extract_embeddings(
        checkpoint=args.checkpoint,
        adata_path=args.adata_path,
        output_path=args.output_path,
        lmdb_dir=args.lmdb_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
