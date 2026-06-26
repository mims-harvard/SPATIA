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
from scprint.model.model_spatial import scPrint
from scprint.tasks.cell_emb_spatial import Embedder
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
}

effective_dataset_name = config.dataset_name
if effective_dataset_name is None:
    effective_dataset_name = config.dataset_path.split("/")[-1].replace(".h5ad", "")
    logger.warning(f"config.dataset_name was not provided, derived '{effective_dataset_name}' from dataset_path.")

logger.info(f"Processing dataset: {effective_dataset_name} from path: {config.dataset_path} with global seed: {config.seed}")
logger.info(f"This script will iterate through all models defined in tag2wandbid.")

logger.info(f"Loading dataset from {config.dataset_path}...")
adata_original = sc.read(config.dataset_path)
logger.info(f"Dataset loaded, original shape: {adata_original.shape}")

adata_for_preprocessing = adata_original.raw.to_adata() if adata_original.raw is not None else adata_original.copy()
logger.info(f"Dataset shape after .raw.to_adata() (if any): {adata_for_preprocessing.shape}")

logger.info(f"Adding .obs annotations...")
adata_for_preprocessing.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
adata_for_preprocessing.obs["assay_ontology_term_id"] = "EFO:0022615"
adata_for_preprocessing.obs["self_reported_ethnicity_ontology_term_id"] = "unknown"
adata_for_preprocessing.obs["sex_ontology_term_id"] = "unknown"
adata_for_preprocessing.obs["donor_id"] = effective_dataset_name
adata_for_preprocessing.obs["cell_type_ontology_term_id"] = "unknown"
adata_for_preprocessing.obs["tissue_ontology_term_id"] = random.choice(list(tissue_mapping.values()))
adata_for_preprocessing.obs["disease_ontology_term_id"] = random.choice(list(disease_mapping.values()))
adata_for_preprocessing.obs["index"] = adata_for_preprocessing.obs.index.astype(str)
adata_for_preprocessing.obs["dataset_name"] = effective_dataset_name
adata_for_preprocessing.obs["development_stage_ontology_term_id"] = "HsapDv:0000266"

logger.info(f"Example obs names: {adata_for_preprocessing.obs_names[:5].tolist()}")
logger.info(f"Example var names: {adata_for_preprocessing.var_names[:5].tolist()}")
logger.info(f"Obs columns: {adata_for_preprocessing.obs.columns.tolist()}")
logger.info(f"Var columns: {adata_for_preprocessing.var.columns.tolist()}")

preprocessor = Preprocessor(
    do_postp=False,
    force_preprocess=True,
    min_nnz_genes=3,
    min_valid_genes_id=100,
)
logger.info(f"Applying Preprocessor...")
adata_fully_preprocessed = preprocessor(adata_for_preprocessing.copy())
logger.info(f"Dataset shape after Preprocessor: {adata_fully_preprocessed.shape}")

embedder_instance = Embedder(
    how="random expr",
    max_len=1000,
    add_zero_genes=0,
    num_workers=16,
    pred_embedding=["cell_type_ontology_term_id"],
    keep_all_cls_pred=False,
    output_expression="none",
    doclass=False,
    fix_missing_image=True,
)

for current_tag, current_wandb_id in tag2wandbid.items():
    model_name_for_loop = f"0303_spatialfm_crossattention_clip_tag{current_tag}"

    current_log_dir = f"log/{model_name_for_loop}"
    os.makedirs(current_log_dir, exist_ok=True)
    current_log_path = f"{current_log_dir}/{effective_dataset_name}_{config.seed}.log"
    
    logger.remove()
    logger.add(sys.stderr, format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{level}] {message}", level="INFO")
    logger.add(current_log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{level}] {message}", level="INFO")

    logger.info(f"--- Processing for Model Tag: {current_tag} (W&B ID: {current_wandb_id}) ---")
    logger.info(f"Dataset: {effective_dataset_name}, Seed: {config.seed}")
    logger.info(f"Log file for this model tag: {current_log_path}")

    checkpoint_path_for_loop = f"data/log/scprint/{current_wandb_id}/checkpoints/epoch=0-step=8132.fix.ckpt"
    if not os.path.exists(checkpoint_path_for_loop):
        checkpoints_folder = f"data/log/scprint/{current_wandb_id}/checkpoints/"
        if not os.path.isdir(checkpoints_folder):
            logger.warning(f"Checkpoints folder not found: {checkpoints_folder} for tag {current_tag}. Skipping this model.")
            continue
        try:
            checkpoint_path_for_loop = get_latest_ckpt(checkpoints_folder)
            logger.info(f"Using latest checkpoint: {checkpoint_path_for_loop}")
        except ValueError as e:
            logger.warning(f"Could not find checkpoint for tag {current_tag} in {checkpoints_folder}: {e}. Skipping this model.")
            continue
    else:
        logger.info(f"Using fixed checkpoint: {checkpoint_path_for_loop}")

    current_embedding_path = f"../embedding/{config.seed}/{model_name_for_loop}/{effective_dataset_name}.npy"
    if os.path.exists(current_embedding_path) and not config.dry_run:
        logger.info(f"Embedding already exists at {current_embedding_path}. Skipping.")
        continue

    logger.info(f"Loading model from checkpoint: {checkpoint_path_for_loop}")
    try:
        model_instance = scPrint.load_from_checkpoint(checkpoint_path_for_loop, precpt_gene_emb=None)
        model_instance.embs = None
    except Exception as e:
        logger.error(f"Failed to load model for tag {current_tag} from {checkpoint_path_for_loop}: {e}")
        logger.error(traceback.format_exc())
        continue

    logger.info(f"Starting embedding generation using model tag {current_tag}...")
    try:
        n_adata_loop, metrics_loop = embedder_instance(
            model_instance,
            adata_fully_preprocessed.copy(), 
            spatial_datadir=config.spatial_datadir,
            image_preprocesser=model_instance.clip_model_type,
            cache=False,
        )
    except Exception as e:
        logger.error(f"Failed during embedding generation for tag {current_tag}: {e}")
        logger.error(traceback.format_exc())
        continue

    if config.dry_run:
        logger.info(f"Dry run: Would save embedding for tag {current_tag} to {current_embedding_path}")
        continue

    embedding_result = n_adata_loop.obsm["scprint"]
    if embedding_result.shape[0] != adata_fully_preprocessed.shape[0]:
        logger.error(f"Embedding shape mismatch for tag {current_tag}! Expected {adata_fully_preprocessed.shape[0]} cells, got {embedding_result.shape[0]}. Skipping save.")
        continue
        
    logger.info(f"Embedding generated for tag {current_tag}, shape: {embedding_result.shape}")

    try:
        os.makedirs(os.path.dirname(current_embedding_path), exist_ok=True)
        np.save(current_embedding_path, embedding_result)
        logger.info(f"Embedding successfully saved to {current_embedding_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding for tag {current_tag} to {current_embedding_path}: {e}")
        logger.error(traceback.format_exc())
    
    logger.info(f"--- Finished processing for Model Tag: {current_tag} ---")

logger.info("=== All model tags processed for this dataset and seed ===")
