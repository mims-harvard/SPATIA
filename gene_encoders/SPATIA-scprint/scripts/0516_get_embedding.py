import os
import pickle
import random
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import *

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tyro
from loguru import logger
from scdataloader import Preprocessor
from scdataloader.utils import get_descendants, load_genes, translate
from scib_metrics.benchmark import Benchmarker
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from utils import cell_mapping, disease_mapping, tissue_mapping


def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_latest_ckpt(checkpoint_dir: str):
    ckpt_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".ckpt")
    ]
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    latest_ckpt = max(ckpt_files, key=os.path.getctime)
    return latest_ckpt


@dataclass(frozen=True)
class CLIArguments:
    tag: str = "1e-0"
    """Tag for the model."""

    dataset_path: str = "dataset_raw/Breast_Cancer_Rep1.h5ad"
    """Name of the dataset."""

    dataset_name: str | None = None
    """Name of the dataset."""

    seed: int = 0
    """Random seed for reproducibility."""

    spatial_datadir: str = "../dataset/lmdb/xenium_multiscale_0514.lmdb"
    """Path to the spatial data directory."""

    dry_run: bool = False
    """If True, only run the embedding generation on some samples without saving the results."""


config = tyro.cli(CLIArguments)
seed_all(config.seed)
tag = config.tag

torch.set_float32_matmul_precision("medium")

tag2wandbid = {
    "1e-07_1e-07_0.1_gfs3ryar": "your_wandb_run_id",
    "1e-08_1e-08_0.1_dhqaztzi": "your_wandb_run_id",
    "0.0001_0.0001_0.1_nd6quwf8": "your_wandb_run_id",
    "1_1_0.1_224ntft7": "your_wandb_run_id",
    "0.1_0.1_0.1_82wbs6f8": "your_wandb_run_id",
    "0.001_0.001_0.1_qghmt7uw": "your_wandb_run_id",
    "0_0_0.1_wul2labn": "your_wandb_run_id",
    "1e-05_1e-05_0.1_2g4wunnp": "your_wandb_run_id",
    "1e-07_1e-07_0.1_sp4byr2t": "your_wandb_run_id",
    "1e-06_1e-06_0.1_lyl47dp4": "your_wandb_run_id",
    "1e-08_1e-08_0.1_jcz7nq8b": "your_wandb_run_id",
    "1e-08_1e-08_0_dx0yhwro": "your_wandb_run_id",
    "0.01_0.01_0.1_ou30tvuk": "your_wandb_run_id",
    "0308_weight0.0_8gr83en8": "your_wandb_run_id",
    "scprint": "scprint",
}

wandb_id = tag2wandbid[tag]

if wandb_id != "scprint":
    from scprint.model.model_spatial import scPrint
    from scprint.tasks.cell_emb_spatial import Embedder

    checkpoint_dir = (
        f"data/log/scprint/{wandb_id}/checkpoints/epoch=0-step=8132.fix.ckpt"
    )
else:
    from scprint.model.model import scPrint
    from scprint.tasks.cell_emb import Embedder

    checkpoint_dir = "data/model/scPRINT/medium.ckpt"
if not os.path.exists(checkpoint_dir):
    checkpoint_dir = f"data/log/scprint/{wandb_id}/checkpoints/"
    checkpoint_dir = get_latest_ckpt(checkpoint_dir)
    logger.info(f"Latest checkpoint: {checkpoint_dir}")
model_name = f"0303_spatialfm_crossattention_clip_tag{tag}"
if config.dataset_name is None:
    dataset_name = config.dataset_path.split("/")[-1].split(".")[0]
else:
    dataset_name = config.dataset_name

log_dir = f"log/{model_name}"
os.makedirs(log_dir, exist_ok=True)
log_path = f"{log_dir}/{dataset_name}_{config.seed}.log"

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(log_path, level="INFO")

logger.info(f"Starting embedding generation for {dataset_name} with seed {config.seed}")
logger.info(f"Using model tag: {tag}, wandb_id: {wandb_id}")
logger.info(f"Checkpoint directory: {checkpoint_dir}")

logger.info("Loading model from checkpoint")
model = scPrint.load_from_checkpoint(checkpoint_dir, precpt_gene_emb=None)
model.embs = None

logger.info(f"Loading dataset from {config.dataset_path}")
adata = sc.read(config.dataset_path)
logger.info(f"Dataset loaded, shape: {adata.shape}")
old_cell_count = adata.shape[0]

adata = adata.raw.to_adata() if adata.raw is not None else adata
assert (
    adata.shape[0] == old_cell_count
), f"Shape mismatch after loading raw data, found {adata.shape[0]}, expected {old_cell_count}"
print(f"Example obs names: {adata.obs_names[:5]}")
print(f"Example var names: {adata.var_names[:5]}")
print(f"Obs columns: {adata.obs.columns}")
print(f"Var columns: {adata.var.columns}")

simple_dataset_name = dataset_name.replace("_filtered", "")
if "_grid_" in simple_dataset_name:
    simple_dataset_name = simple_dataset_name.split("_grid_")[0]
logger.info(
    f"Simple dataset name: {simple_dataset_name}, data set path: {config.dataset_path}"
)

adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
adata.obs["assay_ontology_term_id"] = "EFO:0022615"
adata.obs["self_reported_ethnicity_ontology_term_id"] = "unknown"
adata.obs["sex_ontology_term_id"] = "unknown"
adata.obs["donor_id"] = dataset_name
adata.obs["cell_type_ontology_term_id"] = "unknown"
adata.obs["tissue_ontology_term_id"] = random.choice(list(tissue_mapping.values()))
adata.obs["disease_ontology_term_id"] = random.choice(list(disease_mapping.values()))
adata.obs["index"] = adata.obs.index.astype(str)
adata.obs["dataset_name"] = simple_dataset_name
adata.obs["development_stage_ontology_term_id"] = "HsapDv:0000266"
adata.var_names_make_unique()
preprocessor = Preprocessor(
    do_postp=False,
    force_preprocess=True,
    min_nnz_genes=None,
    min_valid_genes_id=30,
)
adata = preprocessor(adata)
if adata.shape[0] < old_cell_count:
    raise ValueError(
        f"Filtered out {old_cell_count - adata.shape[0]} cells with too few genes"
    )

kwargs = {
    "how": "random expr",
    "max_len": 1000,
    "add_zero_genes": 0,
    "num_workers": 16,
    "pred_embedding": ["cell_type_ontology_term_id"],
    "keep_all_cls_pred": False,
    "output_expression": "none",
    "doclass": False,
    "fix_missing_image": True,
}
if wandb_id == "scprint":
    del kwargs["fix_missing_image"]
embed = Embedder(**kwargs)

logger.info("Starting embedding generation")
kwargs = {
    "model": model,
    "adata": adata.copy(),
    "spatial_datadir": config.spatial_datadir,
    "cache": False,
}
if wandb_id == "scprint":
    del kwargs["spatial_datadir"]
else:
    kwargs["image_preprocesser"] = model.clip_model_type
n_adata, metrics = embed(**kwargs)

embedding = n_adata.obsm["scprint"]
assert (
    embedding.shape[0] == adata.shape[0]
), f"Shape mismatch in embedding, found {embedding.shape[0]}, expected {adata.shape[0]}"
logger.info(f"Embedding generated, shape: {embedding.shape}")

embedding_path = f"../embedding/{config.seed}/{model_name}/{dataset_name}.npy"
os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
np.save(embedding_path, embedding)
logger.info(f"Embedding saved to {embedding_path}")


