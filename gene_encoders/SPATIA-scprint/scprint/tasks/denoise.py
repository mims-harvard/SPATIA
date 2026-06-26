import os
from typing import Any, List, Optional, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import sklearn.metrics
import torch
from anndata import AnnData
from scdataloader import Collator, Preprocessor
from scdataloader.data import SimpleAnnDataset
from scipy.sparse import issparse
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from scprint.model import utils

from . import knn_smooth

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class Denoiser:
    def __init__(
        self,
        batch_size: int = 10,
        num_workers: int = 1,
        max_len: int = 5_000,
        precision: str = "16-mixed",
        how: str = "most var",
        max_cells: int = 500_000,
        doplot: bool = False,
        predict_depth_mult: int = 4,
        downsample: Optional[float] = None,
        dtype: torch.dtype = torch.float16,
        genelist: Optional[List[str]] = None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.max_cells = max_cells
        self.doplot = doplot
        self.predict_depth_mult = predict_depth_mult
        self.how = how
        self.downsample = downsample
        self.precision = precision
        self.dtype = dtype
        self.genelist = genelist

    def __call__(self, model: torch.nn.Module, adata: AnnData):
        if os.path.exists("collator_output.txt"):
            os.remove("collator_output.txt")
        random_indices = None
        if self.max_cells < adata.shape[0]:
            random_indices = np.random.randint(
                low=0, high=adata.shape[0], size=self.max_cells
            )
            adataset = SimpleAnnDataset(
                adata[random_indices], obs_to_output=["organism_ontology_term_id"]
            )
        else:
            adataset = SimpleAnnDataset(
                adata, obs_to_output=["organism_ontology_term_id"]
            )
        if self.how == "most var":
            sc.pp.highly_variable_genes(
                adata, flavor="seurat_v3", n_top_genes=self.max_len, span=0.99
            )
            self.genelist = adata.var.index[adata.var.highly_variable]
            print(len(self.genelist))

        col = Collator(
            organisms=model.organisms,
            valid_genes=model.genes,
            max_len=self.max_len,
            how="some" if self.how == "most var" else self.how,
            genelist=self.genelist if self.how != "random expr" else [],
            downsample=self.downsample,
            save_output=True,
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        model.doplot = self.doplot
        model.on_predict_epoch_start()
        model.eval()
        device = model.device.type
        with torch.no_grad(), torch.autocast(device_type=device, dtype=self.dtype):
            for batch in tqdm(dataloader):
                gene_pos, expression, depth = (
                    batch["genes"].to(device),
                    batch["x"].to(device),
                    batch["depth"].to(device),
                )
                model._predict(
                    gene_pos,
                    expression,
                    depth,
                    predict_mode="denoise",
                    depth_mult=self.predict_depth_mult,
                )
        torch.cuda.empty_cache()
        self.genes = (
            model.pos.cpu().numpy()
            if self.how != "most var"
            else list(set(model.genes) & set(self.genelist))
        )
        tokeep = None
        metrics = None
        if self.downsample is not None:
            reco = model.expr_pred[0]
            reco = reco.cpu().numpy()
            tokeep = np.isnan(reco).sum(1) == 0
            reco = reco[tokeep]
            noisy = np.loadtxt("collator_output.txt")[tokeep]

            if random_indices is not None:
                true = adata.X[random_indices][tokeep]
            else:
                true = adata.X[tokeep]
            true = true.toarray() if issparse(true) else true
            if self.how == "most var":
                true = true[:, adata.var.index.isin(self.genes)]
            else:
                true = np.vstack(
                    [
                        true[
                            i,
                            adata.var.index.get_indexer(np.array(model.genes)[val]),
                        ].copy()
                        for i, val in enumerate(self.genes)
                    ]
                )
            corr_coef, p_value = spearmanr(
                np.vstack([reco[true != 0], noisy[true != 0], true[true != 0]]).T
            )
            metrics = {
                "reco2noisy": corr_coef[0, 1],
                "reco2full": corr_coef[0, 2],
                "noisy2full": corr_coef[1, 2],
            }
        return (
            metrics,
            (
                random_indices[tokeep]
                if random_indices is not None and tokeep is not None
                else random_indices
            ),
            self.genes,
            (
                model.expr_pred[0][tokeep].cpu().numpy()
                if tokeep is not None
                else model.expr_pred[0].cpu().numpy()
            ),
        )


def default_benchmark(
    model: Any,
    default_dataset: str = FILE_DIR + "/../../data/r4iCehg3Tw5IbCLiCIbl.h5ad",
    max_len: int = 5000,
):
    adata = sc.read_h5ad(default_dataset)
    denoise = Denoiser(
        batch_size=40,
        max_len=max_len,
        max_cells=10_000,
        doplot=False,
        num_workers=8,
        predict_depth_mult=10,
        downsample=0.7,
    )
    return denoise(model, adata)[0]


def open_benchmark(model):
    adata = sc.read(
        FILE_DIR + "/../../data/pancreas_atlas.h5ad",
        backup_url="https://figshare.com/ndownloader/files/24539828",
    )
    adata = adata[adata.obs.tech == "inDrop1"]

    train, test = split_molecules(adata.layers["counts"].round().astype(int), 0.9)
    is_missing = np.array(train.sum(axis=0) == 0)
    true = adata.copy()
    true.X = test
    adata.layers["counts"] = train
    test = test[:, ~is_missing.flatten()]
    adata = adata[:, ~is_missing.flatten()]

    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    preprocessor = Preprocessor(
        subset_hvg=3000,
        use_layer="counts",
        is_symbol=True,
        force_preprocess=True,
        skip_validate=True,
        do_postp=False,
    )
    nadata = preprocessor(adata.copy())

    denoise = Denoiser(
        batch_size=32,
        max_len=15_800,
        max_cells=10_000,
        doplot=False,
        predict_depth_mult=1.2,
        downsample=None,
    )
    expr = denoise(model, nadata)
    denoised = ad.AnnData(
        expr.cpu().numpy(),
        var=nadata.var.loc[
            np.array(denoise.model.genes)[
                denoise.model.pos[0].cpu().numpy().astype(int)
            ]
        ],
    )
    denoised = denoised[:, denoised.var.symbol.isin(true.var.index)]
    loc = true.var.index.isin(denoised.var.symbol)
    true = true[:, loc]
    train = train[:, loc]

    denoised = denoised[
        :, denoised.var.set_index("symbol").index.get_indexer(true.var.index)
    ]
    denoised.X = np.maximum(denoised.X - train.astype(float), 0)
    target_sum = 1e4

    sc.pp.normalize_total(true, target_sum)
    sc.pp.log1p(true)

    sc.pp.normalize_total(denoised, target_sum)
    sc.pp.log1p(denoised)

    error_mse = sklearn.metrics.mean_squared_error(true.X, denoised.X)
    initial_sum = train.sum()
    target_sum = true.X.sum()
    denoised.X = denoised.X * target_sum / initial_sum
    error_poisson = poisson_nll_loss(true.X, denoised.X)
    return {"error_poisson": error_poisson, "error_mse": error_mse}


def mse(test_data, denoised_data, target_sum=1e4):
    sc.pp.normalize_total(test_data, target_sum)
    sc.pp.log1p(test_data)

    sc.pp.normalize_total(denoised_data, target_sum)
    sc.pp.log1p(denoised_data)

    print("Compute mse value", flush=True)
    return sklearn.metrics.mean_squared_error(test_data.X.todense(), denoised_data.X)


def withknn(adata, k=10, **kwargs):
    adata.layers["denoised"] = knn_smooth.knn_smoothing(
        adata.X.transpose(), k=k, **kwargs
    ).transpose()
    return adata


def poisson_nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()


def split_molecules(
    umis: np.ndarray,
    data_split: float,
    overlap_factor: float = 0.0,
    random_state: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is None:
        random_state = np.random.RandomState()

    umis_X_disjoint = random_state.binomial(umis, data_split - overlap_factor)
    umis_Y_disjoint = random_state.binomial(
        umis - umis_X_disjoint, (1 - data_split) / (1 - data_split + overlap_factor)
    )
    overlap_factor = umis - umis_X_disjoint - umis_Y_disjoint
    umis_X = umis_X_disjoint + overlap_factor
    umis_Y = umis_Y_disjoint + overlap_factor

    return umis_X, umis_Y
