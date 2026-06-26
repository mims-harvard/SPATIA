import json
import os
import sys
from pathlib import Path

import torch
from loguru import logger

from ..data_collator import DataCollator
from ..tokenizer import GeneVocab

from .dataset import MultiAdataDataset, create_dataloaders

BASE_DATA_DIR = Path(os.environ.get("SPATIA_DATA_DIR", "./data"))
MODEL_CKPT_DIR = Path(os.environ.get("SPATIA_CHECKPOINT_DIR", "./checkpoints"))
VOCAB_FILE = MODEL_CKPT_DIR / "vocab.json"
MODEL_CONFIG_FILE = MODEL_CKPT_DIR / "args.json"
GENE_STATS_FILE = MODEL_CKPT_DIR / "all_dict_mean_std.csv"
ADATA_FILE_NAME = "adata_processed_0212.h5ad"

DATASET_NAMES = [
    "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs",
    "Xenium_FFPE_Human_Breast_Cancer_Rep2_outs",
    "Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs",
]


def test_real_data_loading_and_processing():
    logger.info("Starting real data loading and processing test...")

    if not VOCAB_FILE.exists():
        logger.error(f"Vocab file not found: {VOCAB_FILE}")
        raise FileNotFoundError(f"Vocab file not found: {VOCAB_FILE}")
    if not MODEL_CONFIG_FILE.exists():
        logger.error(f"Model config file not found: {MODEL_CONFIG_FILE}")
        raise FileNotFoundError(f"Model config file not found: {MODEL_CONFIG_FILE}")
    if not GENE_STATS_FILE.exists():
        logger.error(f"Gene stats file not found: {GENE_STATS_FILE}")
        raise FileNotFoundError(f"Gene stats file not found: {GENE_STATS_FILE}")

    vocab = GeneVocab.from_file(VOCAB_FILE)
    logger.info(f"Loaded vocabulary with {len(vocab)} tokens.")

    with open(MODEL_CONFIG_FILE, "r") as f:
        model_configs = json.load(f)
    logger.info(f"Loaded model configurations: {list(model_configs.keys())}")

    adata_paths = [BASE_DATA_DIR / name / ADATA_FILE_NAME for name in DATASET_NAMES]

    existing_adata_paths = [p for p in adata_paths if p.exists()]
    if len(existing_adata_paths) < len(adata_paths):
        logger.warning(
            f"Found {len(existing_adata_paths)} out of {len(adata_paths)} expected adata files. "
            f"Missing files will be skipped."
        )
        if not existing_adata_paths:
            logger.error("No adata files found at the specified paths. Aborting test.")
            return

    logger.info(f"Attempting to load {len(existing_adata_paths)} AnnData files.")

    gene_col = model_configs.get(
        "gene_col", "gene_name"
    )
    cls_token = model_configs.get("cls_token", "<cls>")
    expression_pad_value = model_configs.get("pad_value", 0.0)
    require_log1p = model_configs.get("require_log1p", True)

    try:
        multi_dataset = MultiAdataDataset(
            adata_paths=existing_adata_paths,
            vocab=vocab,
            gene_stats_file=GENE_STATS_FILE,
            gene_col=gene_col,
            cls_token=cls_token,
            pad_value=expression_pad_value,
            require_log1p=require_log1p,
        )
    except Exception as e:
        logger.error(f"Error initializing MultiAdataDataset: {e}")
        raise

    logger.info(
        f"Successfully loaded MultiAdataDataset with {len(multi_dataset)} total cells."
    )
    assert len(multi_dataset) > 0, "MultiAdataDataset is empty after loading."
    logger.info(
        "All specified AnnData files were loaded and validated by MultiAdataDataset."
    )

    max_length = model_configs.get("max_seq_len", model_configs.get("max_length", 1200))
    pad_token = model_configs.get("pad_token", "<pad>")

    if pad_token not in vocab:
        logger.warning(
            f"Pad token '{pad_token}' not explicitly in vocab. Relying on default index if any."
        )
        pad_token_id = vocab.get(
            pad_token, vocab.default_index if hasattr(vocab, "default_index") else 0
        )
    else:
        pad_token_id = vocab[pad_token]

    do_binning = model_configs.get("do_binning", True)
    n_bins = model_configs.get("n_bins", 51)

    try:
        data_collator = DataCollator(
            pad_token_id=pad_token_id,
            pad_value=expression_pad_value,
            do_mlm=model_configs.get(
                "do_mlm", False
            ),
            do_binning=do_binning,
            n_bins=n_bins,
            max_length=max_length,
            do_padding=True,
            sampling=model_configs.get("sampling", True),
            keep_first_n_tokens=model_configs.get(
                "keep_first_n_tokens", 1
            ),
        )
        logger.info(
            f"DataCollator initialized. Max length: {max_length}, Binning: {do_binning}, N_bins: {n_bins if do_binning else 'N/A'}."
        )
    except TypeError as te:
        logger.error(
            f"TypeError initializing DataCollator. This might be due to mismatched arguments "
            f"for the imported scgpt_spatial.data_collator.DataCollator: {te}"
        )
        logger.error(
            "Please ensure the DataCollator in scgpt_spatial.data_collator.py "
            "accepts or defaults parameters like: pad_token_id, pad_value, do_mlm, "
            "do_binning, n_bins, max_length, do_padding, sampling, keep_first_n_tokens."
        )
        raise
    except Exception as e:
        logger.error(f"Unknown error initializing DataCollator: {e}")
        raise

    batch_size = model_configs.get(
        "batch_size", 4
    )
    validation_split = 0.1

    train_loader, val_loader = create_dataloaders(
        dataset=multi_dataset,
        data_collator=data_collator,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        num_workers=0,
    )

    logger.info(f"Created train_loader and val_loader. Batch size: {batch_size}.")

    logger.info("Testing train_loader...")
    train_batches_to_check = min(2, len(train_loader) if train_loader else 0)
    if not train_loader or train_batches_to_check == 0:
        logger.warning("Train loader is empty or too small to test.")
    else:
        for i, batch in enumerate(train_loader):
            if i >= train_batches_to_check:
                break
            logger.info(f"Train batch {i+1}/{train_batches_to_check}:")
            logger.info(f"  Batch keys: {list(batch.keys())}")

            gene_key = None
            for key in batch.keys():
                if "gene" in key.lower():
                    gene_key = key
                    break
            if gene_key is None:
                logger.error(
                    "Train batch missing 'gene' key. Check your dataset and DataCollator."
                )
                raise KeyError("Train batch missing 'gene' key.")
            expression_key = None
            for key in batch.keys():
                if "expr" in key.lower():
                    expression_key = key
                    break
            if expression_key is None:
                logger.error(
                    "Train batch missing 'expressions' key. Check your dataset and DataCollator."
                )
                raise KeyError("Train batch missing 'expressions' key.")

            genes = batch[gene_key]
            expressions = batch[expression_key]

            logger.info(f"  genes shape: {genes.shape}, dtype: {genes.dtype}")
            logger.info(
                f"  expressions shape: {expressions.shape}, dtype: {expressions.dtype}"
            )

            assert genes.ndim == 2, f"genes tensor should be 2D, got {genes.ndim}D"
            assert (
                expressions.ndim == 2
            ), f"expressions tensor should be 2D, got {expressions.ndim}D"
            assert (
                genes.shape[0] <= batch_size
            ), f"genes batch size incorrect: expected <= {batch_size}, got {genes.shape[0]}"
            assert (
                expressions.shape[0] <= batch_size
            ), f"expressions batch size incorrect: expected <= {batch_size}, got {expressions.shape[0]}"
            assert (
                genes.dtype == torch.long
            ), f"genes dtype should be torch.long, got {genes.dtype}"

            expected_expr_dtype = torch.long if do_binning else torch.float
            expected_expr_dtype = torch.float
            assert (
                expressions.dtype == expected_expr_dtype
            ), f"expressions dtype incorrect: expected {expected_expr_dtype}, got {expressions.dtype}"

            if (
                "batch_labels" in batch
            ):
                logger.info(
                    f"  batch_labels shape: {batch['batch_labels'].shape}, dtype: {batch['batch_labels'].dtype}"
                )
                assert batch["batch_labels"].dtype == torch.long

    if val_loader:
        logger.info("Testing val_loader...")
        val_batches_to_check = min(2, len(val_loader))
        if val_batches_to_check == 0:
            logger.warning("Val loader is empty or too small to test.")
        else:
            for i, batch in enumerate(val_loader):
                if i >= val_batches_to_check:
                    break
                logger.info(f"Val batch {i+1}/{val_batches_to_check}:")
                gene_key = None
                for key in batch.keys():
                    if "gene" in key.lower():
                        gene_key = key
                        break
                if gene_key is None:
                    logger.error(
                        "Train batch missing 'gene' key. Check your dataset and DataCollator."
                    )
                    raise KeyError("Train batch missing 'gene' key.")
                expression_key = None
                for key in batch.keys():
                    if "expr" in key.lower():
                        expression_key = key
                        break
                if expression_key is None:
                    logger.error(
                        "Train batch missing 'expressions' key. Check your dataset and DataCollator."
                    )
                    raise KeyError("Train batch missing 'expressions' key.")

                genes = batch[gene_key]
                expressions = batch[expression_key]

                logger.info(f"  genes shape: {genes.shape}, dtype: {genes.dtype}")
                logger.info(
                    f"  expressions shape: {expressions.shape}, dtype: {expressions.dtype}"
                )

                assert genes.ndim == 2
                assert expressions.ndim == 2
                assert genes.shape[0] <= batch_size
                assert expressions.shape[0] <= batch_size
                assert genes.dtype == torch.long

                expected_expr_dtype = torch.long if do_binning else torch.float
                expected_expr_dtype = torch.float
                assert (
                    expressions.dtype == expected_expr_dtype
                ), f"expressions dtype incorrect: expected {expected_expr_dtype}, got {expressions.dtype}"

                if "batch_labels" in batch:
                    logger.info(
                        f"  batch_labels shape: {batch['batch_labels'].shape}, dtype: {batch['batch_labels'].dtype}"
                    )
                    assert batch["batch_labels"].dtype == torch.long
    else:
        logger.info(
            "No validation loader to test (validation_split was 0 or dataset too small)."
        )

    logger.info("Dataloader tests passed successfully!")


if __name__ == "__main__":

    logger.info("Running test_dataset.py as a script.")
    try:
        test_real_data_loading_and_processing()
    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
        logger.error(
            "Please ensure all paths (data, vocab, config, gene_stats) are correct and files exist."
        )
    except Exception as e:
        logger.exception(f"An error occurred during the test run: {e}")

