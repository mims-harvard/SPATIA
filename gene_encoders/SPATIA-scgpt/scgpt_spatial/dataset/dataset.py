
import os
from math import e
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata
import lmdb
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from loguru import logger
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
    SequentialSampler,
    random_split,
)
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

from ..data_collator import DataCollator
from ..tokenizer import GeneVocab

PathLike = Union[str, os.PathLike]


def _open_lmdb(lmdb_path: str) -> lmdb.Environment:
    return lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        map_size=1024**4 * 4,
    )


class SingleAdataDataset(Dataset):

    def __init__(
        self,
        adata_path: PathLike,
        vocab: GeneVocab,
        gene_stats_file: PathLike,
        gene_col: str,
        cls_token: str = "<cls>",
        pad_value: float = 0.0,
        require_log1p: bool = True,
        spatial_datadir: Optional[str] = None,
        fix_missing_images: bool = False,
        preprocessor_cls: Optional[str] = None,
        inference: bool = False,
        cell_type_col: str = "auto",
    ):
        super().__init__()
        self.adata_path = Path(adata_path)
        self.vocab = vocab
        self.gene_stats_file = Path(gene_stats_file)
        self.gene_col = gene_col
        self.cls_token = cls_token
        self.cls_token_id = self.vocab[self.cls_token]
        self.pad_value = pad_value
        self.require_log1p = require_log1p
        self.spatial_datadir = spatial_datadir
        self.fix_missing_images = fix_missing_images
        self.preprocessor_cls = preprocessor_cls
        self.inference = inference
        self._lmdb_env = None
        if self.preprocessor_cls is not None:
            self.image_preprocessor = AutoImageProcessor.from_pretrained(
                self.preprocessor_cls
            )
        self.cell_types = set()
        self.cell_type_col = cell_type_col
        self.num_cell_types = 0

        self._load_and_process_data()

    @property
    def env(self):
        if self._lmdb_env is None and self.spatial_datadir is not None:
            self._lmdb_env = _open_lmdb(self.spatial_datadir)
        return self._lmdb_env

    def _validate_adata(self, adata: anndata.AnnData) -> anndata.AnnData:
        logger.info(f"Validating adata: {self.adata_path.name}")

        if self.require_log1p:
            if adata.X.max() > 1000:
                logger.warning(
                    f"Adata {self.adata_path.name} does not appear to be log1p transformed. "
                    "Expected `adata.uns['log1p']` to exist."
                )
            else:
                logger.info("Normalization check passed (log1p found).")

        if self.gene_col == "index" and "index" not in adata.var.columns:
            adata.var["index"] = adata.var_names

        if self.gene_col not in adata.var.columns:
            raise ValueError(
                f"Required gene column '{self.gene_col}' not found in adata.var. "
                f"Available columns: {list(adata.var.columns)}"
            )
        logger.info(f"Gene column '{self.gene_col}' found.")

        sample_genes = adata.var[self.gene_col].head(10)
        if all(str(g).startswith("ENSG") for g in sample_genes):
            logger.warning(
                f"Gene names in '{self.gene_col}' appear to be Ensembl IDs. "
                "The model typically works best with gene symbols."
            )

        if not self.inference:
            n_cells_before, n_genes_before = adata.shape
            sc.pp.filter_genes(adata, min_cells=1)
            sc.pp.filter_cells(adata, min_genes=1)
            n_cells_after, n_genes_after = adata.shape
            if (n_cells_before, n_genes_before) != (n_cells_after, n_genes_after):
                logger.info(
                    f"Filtered {n_cells_before - n_cells_after} cells and "
                    f"{n_genes_before - n_genes_after} genes with zero counts."
                )

        return adata

    def _load_and_process_data(self):
        logger.info(f"Loading data from {self.adata_path}...")
        adata = sc.read_h5ad(self.adata_path)

        adata = self._validate_adata(adata)

        original_gene_count = adata.n_vars
        adata.var["id_in_vocab"] = [self.vocab[g] for g in adata.var[self.gene_col]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
        logger.info(
            f"Filtered genes to {adata.n_vars}/{original_gene_count} "
            f"present in the vocabulary."
        )

        self.adata = adata
        self.gene_ids = self.adata.var["id_in_vocab"].values

        self.count_matrix = self.adata.X
        if not isinstance(self.count_matrix, np.ndarray):
            self.count_matrix = self.count_matrix.toarray()
        logger.info(f"Count matrix shape: {self.count_matrix.shape}")

        self.slide_mean = np.mean(self.count_matrix[self.count_matrix.nonzero()])
        self.count_matrix = self.count_matrix / self.slide_mean

        self.gene_stats_dict = pd.read_csv(self.gene_stats_file, index_col=0)

        new_genes_mask = ~np.isin(self.gene_ids, self.gene_stats_dict.index.values)
        if np.any(new_genes_mask):
            new_gene_ids = self.gene_ids[new_genes_mask]
            logger.info(
                f"Found {len(new_gene_ids)} new genes not in stats file. Calculating their means..."
            )
            for gene_id in np.unique(new_gene_ids):
                gene_col_indices = np.where(self.gene_ids == gene_id)[0]
                gene_columns = self.count_matrix[:, gene_col_indices]
                values = gene_columns[np.nonzero(gene_columns)]
                gene_mean = (
                    float(values.mean()) if values.size > 0 else 1.0
                )
                self.gene_stats_dict.loc[gene_id] = [gene_mean]

        self._gene_mean_dict = self.gene_stats_dict["mean"].to_dict()

        if self.spatial_datadir is not None:
            self.image_keys = []
            has_ds_col = "dataset_name" in adata.obs.columns
            has_idx_col = "index" in adata.obs.columns
            for i in range(len(self.adata)):
                if has_ds_col:
                    dataset_name = adata.obs["dataset_name"].iloc[i].split("/")[-1]
                    if dataset_name.endswith(".h5ad"):
                        dataset_name = dataset_name.replace(".h5ad", "")
                else:
                    dataset_name = Path(self.spatial_datadir).stem
                cell_id = (
                    adata.obs["index"].iloc[i]
                    if has_idx_col
                    else str(adata.obs.index[i])
                )
                if isinstance(cell_id, bytes):
                    cell_id = cell_id.decode("utf-8")
                image_key = f"{dataset_name}/{cell_id}"
                self.image_keys.append(image_key)

        if self.cell_type_col == "auto":
            potential_cols = ["cell_type", "celltype", "annotation"]
            for col in potential_cols:
                if col in self.adata.obs.columns:
                    self.cell_type_col = col
                    break
        if self.cell_type_col in self.adata.obs.columns:
            self.cell_types = set(self.adata.obs[self.cell_type_col].unique()) | {
                "unknown"
            }
            if len(self.cell_types) == 0:
                logger.warning(
                    f"No unique cell types found in column '{self.cell_type_col}'. "
                    "Ensure that the column is populated."
                )
                self.cell_types = {"unknown"}
            self.cell_types = list(self.cell_types)
            self.num_cell_types = len(self.cell_types)
            logger.info(
                f"Found {self.num_cell_types} unique cell types in column '{self.cell_type_col}'."
            )
        elif self.cell_type_col is not None:
            raise ValueError(
                f"Cell type column '{self.cell_type_col}' not found in adata.obs with file path {self.adata_path}. "
                f"Available columns: {list(self.adata.obs.columns)}"
            )
        else:
            logger.warning(
                "No cell type column specified or found. "
                "Cell types will not be available in the dataset."
            )
            self.cell_types = ["unknown"]
            self.num_cell_types = 1

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]

        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]

        gene_means = np.array([self._gene_mean_dict.get(g, 1.0) for g in genes])
        values = np.divide(values, gene_means + 1e-8)

        genes = np.insert(genes, 0, self.cls_token_id)
        values = np.insert(values, 0, self.pad_value)
        image = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.spatial_datadir is not None:
            image_key = self.image_keys[idx]
            with self.env.begin() as txn:
                key = image_key.encode("utf-8")
                value = txn.get(key)
                if value is None:
                    if self.fix_missing_images:
                        logger.warning(
                            f"Image for key {image_key} not found, using placeholder."
                        )
                        data = np.zeros((256, 256), dtype=np.uint8)
                    else:
                        logger.error(
                            f"Image for key {image_key} not found. "
                            "Set `fix_missing_images=True` to use a placeholder."
                        )
                        raise KeyError(f"Image for key {image_key} not found.")
                else:
                    data = np.frombuffer(value, dtype=np.uint8)
                    data = data.reshape(256, 256)
                image = self.image_preprocessor(
                    images=np.stack((data,) * 3, axis=-1),
                    return_tensors="pt",
                )["pixel_values"][0]

        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
            "image": image,
            "cell_type": (
                self.adata.obs[self.cell_type_col].iloc[idx]
                if self.cell_type_col in self.adata.obs.columns
                else "unknown"
            ),
        }


class MultiAdataDataset(Dataset):

    def __init__(self, adata_paths: List[PathLike], *args, **kwargs):
        super().__init__()
        self.datasets = [
            SingleAdataDataset(path, *args, **kwargs)
            for path in tqdm(adata_paths, desc="Loading AnnData files")
        ]
        self.concat_dataset = ConcatDataset(self.datasets)
        self.cell_types = {"unknown"}
        for ds in self.datasets:
            self.cell_types = self.cell_types | set(ds.cell_types)
        if len(self.cell_types) == 0:
            logger.warning(
                "No cell types found in the datasets. "
                "Ensure that `adata.obs['cell_type']` is populated."
            )
            self.cell_types = {"unknown"}
        self.cell_types = list(self.cell_types)
        self.num_cell_types = len(self.cell_types)

        total_cells = sum(len(ds.adata) for ds in self.datasets)
        logger.info(
            f"Created MultiAdataDataset with {len(self.datasets)} files, "
            f"containing a total of {total_cells} cells."
        )

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.concat_dataset[idx]


def create_dataloaders(
    dataset: Dataset,
    data_collator: DataCollator,
    batch_size: int = 64,
    shuffle: bool = True,
    validation_split: Optional[float] = 0.1,
    num_workers: int = 0,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be between 0.0 and 1.0")

    if validation_split > 0.0:
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], **kwargs
        )
        logger.info(
            f"Split dataset into {len(train_dataset)} training samples "
            f"and {len(val_dataset)} validation samples."
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_sampler = (
        torch.utils.data.RandomSampler(train_dataset)
        if shuffle
        else SequentialSampler(train_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(val_dataset),
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
