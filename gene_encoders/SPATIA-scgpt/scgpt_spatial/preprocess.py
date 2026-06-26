from typing import Dict, Optional, Union, Any

import numpy as np
import torch
from scipy.sparse import issparse
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData
import pandas as pd

from scgpt_spatial import logger


class Preprocessor:

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        add_cell_norm: Optional[bool] = False,
        add_gene_norm: Optional[str] = None,
        vocab: Optional[Any] = None,
        result_add_norm_key: Optional[str] = 'X_add_normed',
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.add_cell_norm = add_cell_norm
        self.add_gene_norm = add_gene_norm
        self.vocab = vocab
        self.result_add_norm_key = result_add_norm_key
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        key_to_process = self.use_key
        if key_to_process == "X":
            key_to_process = None
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        if isinstance(self.filter_cell_by_counts, int):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        if self.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        if self.log1p:
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        if self.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                if isinstance(self.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )

        if self.add_cell_norm:
            logger.info("Perform cell-wise norm ...")
            layer_data = adata.X.copy()
            layer_data = layer_data.todense() if issparse(layer_data) else layer_data
            dataset_mean = np.mean(layer_data[layer_data.nonzero()[0], layer_data.nonzero()[1]])
            add_normed_ = layer_data / dataset_mean
            key_to_process = self.result_add_norm_key or key_to_process
            _set_obs_rep(adata, add_normed_, layer=key_to_process)
        if self.add_gene_norm:
            logger.info("Perform population-based gene-wise norm ...")
            gene_stats_dict = pd.read_csv(self.add_gene_norm, index_col=0)
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            gene_ids = np.array(self.vocab(adata.var["gene_name"].tolist()), dtype=int)
            new_genes = set(gene_ids).difference(set(gene_stats_dict.index.values))
            for i in new_genes:
                idx = np.where(gene_ids==i)[0]
                col = layer_data[:, idx].flatten()
                nonzero_idx = np.nonzero(col)[0]
                values = col[nonzero_idx]
                gene_stats_dict.loc[i] = [np.float(values.mean())]
            for i, g in enumerate(gene_ids):
                col = layer_data[:, i]
                nonzero_idx = np.nonzero(col)[0]
                layer_data[nonzero_idx, i] = np.squeeze(layer_data[nonzero_idx, i]) / gene_stats_dict.loc[g, 'mean']
            add_normed_ = np.array(layer_data)
            key_to_process = self.result_add_norm_key or key_to_process
            _set_obs_rep(adata, add_normed_, layer=key_to_process)

        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            for row in layer_data:
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row

    if row.size == 0:
        return torch.zeros(0, dtype=torch.int64) if not return_np else np.zeros(0, dtype=dtype)
    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)