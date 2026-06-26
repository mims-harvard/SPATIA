import os
from typing import Any, Dict, List

import bionty as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from lightning.pytorch import Trainer
from networkx import average_node_connectivity
from scdataloader import Preprocessor
from scdataloader.collator_spatial import Collator
from scdataloader.data_spatial import SimpleAnnDataset
from scdataloader.utils import get_descendants
from scib_metrics.benchmark import Benchmarker
from scipy.stats import spearmanr
from scprint.model import utils
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

FILE_LOC = os.path.dirname(os.path.realpath(__file__))


class Embedder:
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        how: str = "random expr",
        max_len: int = 2000,
        doclass: bool = True,
        add_zero_genes: int = 0,
        precision: str = "16-mixed",
        pred_embedding: List[str] = [
            "cell_type_ontology_term_id",
            "disease_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
        ],
        plot_corr_size: int = 64,
        doplot: bool = True,
        keep_all_cls_pred: bool = False,
        dtype: torch.dtype = torch.float16,
        output_expression: str = "none",
        genelist: List[str] = [],
        fix_missing_image: bool = False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.how = how
        self.max_len = max_len
        self.add_zero_genes = add_zero_genes
        self.pred_embedding = pred_embedding
        self.keep_all_cls_pred = keep_all_cls_pred
        self.plot_corr_size = plot_corr_size
        self.precision = precision
        self.doplot = doplot
        self.dtype = dtype
        self.doclass = doclass
        self.output_expression = output_expression
        self.genelist = genelist
        self.fix_missing_image = fix_missing_image

    def __call__(
        self,
        model: torch.nn.Module,
        adata: AnnData,
        spatial_datadir: str = None,
        image_preprocesser: str = None,
        cache=False,
    ):
        if cache == True:
            raise ValueError("cache is not supported yet")

        try:
            mdir = (
                model.logger.save_dir if model.logger.save_dir is not None else "data"
            )
        except:
            mdir = "data"
        try:
            file = (
                mdir
                + "/step_"
                + str(model.global_step)
                + "_predict_part_"
                + str(model.counter)
                + "_"
                + str(model.global_rank)
                + ".h5ad"
            )
            hasfile = os.path.exists(file)
            print("h5ad file path", file)
        except:
            hasfile = False

        pred_adata = None
        if not cache or not hasfile:
            model.predict_mode = "none"
            model.keep_all_cls_pred = self.keep_all_cls_pred
            if self.how == "most var":
                sc.pp.highly_variable_genes(
                    adata, flavor="seurat_v3", n_top_genes=self.max_len
                )
                self.genelist = adata.var.index[adata.var.highly_variable]
            adataset = SimpleAnnDataset(
                adata,
                obs_to_output=["organism_ontology_term_id"],
                spatial_datadir=spatial_datadir,
                image_preprocesser=image_preprocesser,
                fix_missing_image=self.fix_missing_image,
            )
            col = Collator(
                organisms=model.organisms,
                valid_genes=model.genes,
                how=self.how if self.how != "most var" else "some",
                max_len=self.max_len,
                add_zero_genes=self.add_zero_genes,
                genelist=self.genelist if self.how in ["most var", "some"] else [],
            )
            dataloader = DataLoader(
                adataset,
                collate_fn=col,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            model.eval()
            model.on_predict_epoch_start()
            device = model.device.type
            model.doplot = self.doplot
            with (
                torch.no_grad(),
                torch.autocast(device_type=device, dtype=self.dtype),
            ):
                for batch in tqdm(dataloader):
                    gene_pos, expression, depth, image = (
                        batch["genes"].to(device),
                        batch["x"].to(device),
                        batch["depth"].to(device),
                        batch["image"].to(device),
                    )
                    model._predict(
                        gene_pos,
                        expression,
                        depth,
                        image,
                        predict_mode="none",
                        pred_embedding=self.pred_embedding,
                    )
                    torch.cuda.empty_cache()
            pred_adata = model.get_adata()
            try:
                mdir = (
                    model.logger.save_dir
                    if model.logger.save_dir is not None
                    else "data"
                )
            except:
                mdir = "data"
            file = (
                mdir
                + "/step_"
                + str(model.global_step)
                + "_"
                + model.name
                + "_predict_part_"
                + str(model.counter)
                + "_"
                + str(model.global_rank)
                + ".h5ad"
            )
            print("h5ad file path", file)

        if pred_adata is None:
            print(f"Loading from {file} instead of recomputing")
            pred_adata = sc.read_h5ad(file)
        else:
            print(f"Recomputed pred_adata")
        if self.output_expression == "all":
            adata.obsm["scprint_mu"] = model.expr_pred[0]
            adata.obsm["scprint_theta"] = model.expr_pred[1]
            adata.obsm["scprint_pi"] = model.expr_pred[2]
            adata.obsm["scprint_pos"] = model.pos.cpu().numpy()
        elif self.output_expression == "sample":
            adata.obsm["scprint_expr"] = (
                utils.zinb_sample(
                    model.expr_pred[0],
                    model.expr_pred[1],
                    model.expr_pred[2],
                )
                .cpu()
                .numpy()
            )
            adata.obsm["scprint_pos"] = model.pos.cpu().numpy()
        elif self.output_expression == "old":
            expr = np.array(model.expr_pred[0])
            expr[
                np.random.binomial(
                    1,
                    p=np.array(
                        torch.nn.functional.sigmoid(
                            model.expr_pred[2].to(torch.float32)
                        )
                    ),
                ).astype(bool)
            ] = 0
            expr[expr <= 0.3] = 0
            expr[(expr >= 0.3) & (expr <= 1)] = 1
            adata.obsm["scprint_expr"] = expr.astype(int)
            adata.obsm["scprint_pos"] = model.pos.cpu().numpy()
        else:
            pass
        pred_adata.obs.index = adata.obs.index
        try:
            adata.obsm["scprint_umap"] = pred_adata.obsm["X_umap"]
        except:
            print("too few cells to embed into a umap")
        try:
            adata.obsm["scprint_leiden"] = pred_adata.obsm["leiden"]
        except:
            print("too few cells to compute a clustering")
        adata.obsm["scprint"] = pred_adata.X
        pred_adata.obs.index = adata.obs.index
        adata.obs = pd.concat([adata.obs, pred_adata.obs], axis=1)
        if self.keep_all_cls_pred:
            allclspred = model.pred
            columns = []
            for cl in model.classes:
                n = model.label_counts[cl]
                columns += [model.label_decoders[cl][i] for i in range(n)]
            allclspred = pd.DataFrame(
                allclspred, columns=columns, index=adata.obs.index
            )
            adata.obs = pd.concat(adata.obs, allclspred)

        metrics = {}
        if self.doclass and not self.keep_all_cls_pred:
            for cl in model.classes:
                res = []
                if cl not in adata.obs.columns:
                    continue
                class_topred = model.label_decoders[cl].values()

                if cl in model.labels_hierarchy:
                    cur_labels_hierarchy = {
                        model.label_decoders[cl][k]: [
                            model.label_decoders[cl][i] for i in v
                        ]
                        for k, v in model.labels_hierarchy[cl].items()
                    }
                else:
                    cur_labels_hierarchy = {}

                for pred, true in adata.obs[["pred_" + cl, cl]].values:
                    if pred == true:
                        res.append(True)
                        continue
                    if len(cur_labels_hierarchy) > 0:
                        if true in cur_labels_hierarchy:
                            res.append(pred in cur_labels_hierarchy[true])
                            continue
                        elif true not in class_topred:
                            raise ValueError(
                                f"true label {true} not in available classes"
                            )
                        elif true != "unknown":
                            res.append(False)
                    elif true not in class_topred:
                        raise ValueError(f"true label {true} not in available classes")
                    elif true != "unknown":
                        res.append(False)
                if len(res) == 0:
                    res = [1]
                if self.doplot:
                    print("    ", cl)
                    print("     accuracy:", sum(res) / len(res))
                    print(" ")
                metrics.update({cl + "_accuracy": sum(res) / len(res)})
        return adata, metrics


class LaminEmbedder:
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        how: str = "random expr",
        max_len: int = 2000,
        doclass: bool = True,
        add_zero_genes: int = 0,
        precision: str = "16-mixed",
        pred_embedding: List[str] = [
            "cell_type_ontology_term_id",
            "disease_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
        ],
        plot_corr_size: int = 64,
        doplot: bool = True,
        keep_all_cls_pred: bool = False,
        dtype: torch.dtype = torch.float16,
        output_expression: str = "none",
        genelist: List[str] = [],
        fix_missing_image: bool = False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.how = how
        self.max_len = max_len
        self.add_zero_genes = add_zero_genes
        self.pred_embedding = pred_embedding
        self.keep_all_cls_pred = keep_all_cls_pred
        self.plot_corr_size = plot_corr_size
        self.precision = precision
        self.doplot = doplot
        self.dtype = dtype
        self.doclass = doclass
        self.output_expression = output_expression
        self.genelist = genelist
        self.fix_missing_image = fix_missing_image

    def __call__(
        self,
        model: torch.nn.Module,
        datamodule,
        cache=False,
    ):
        if cache == True:
            raise ValueError("cache is not supported yet")

        try:
            mdir = (
                model.logger.save_dir if model.logger.save_dir is not None else "data"
            )
        except:
            mdir = "data"
        try:
            file = (
                mdir
                + "/step_"
                + str(model.global_step)
                + "_predict_part_"
                + str(model.counter)
                + "_"
                + str(model.global_rank)
                + ".h5ad"
            )
            hasfile = os.path.exists(file)
            print("h5ad file path", file)
        except:
            hasfile = False

        indexes = []
        pred_adata = None
        if not cache or not hasfile:
            model.predict_mode = "none"
            model.keep_all_cls_pred = self.keep_all_cls_pred

            if (
                not hasattr(datamodule, "dataset_is_setup")
                or not datamodule.dataset_is_setup
            ):
                print("Setting up DataModule...")
                datamodule.setup()
                datamodule.dataset_is_setup = True
            else:
                print("DataModule already set up, skipping setup.")

            dataloader = datamodule.train_dataloader()

            model.eval()
            model.on_predict_epoch_start()
            device = model.device.type
            model.doplot = self.doplot
            with (
                torch.no_grad(),
                torch.autocast(device_type=device, dtype=self.dtype),
            ):
                for batch in tqdm(dataloader):
                    gene_pos, expression, depth, image = (
                        batch["genes"].to(device),
                        batch["x"].to(device),
                        batch["depth"].to(device),
                        batch["image"].to(device),
                    )
                    index = batch["index"]
                    indexes.extend(index)
                    model._predict(
                        gene_pos,
                        expression,
                        depth,
                        image,
                        predict_mode="none",
                        pred_embedding=self.pred_embedding,
                        keep_output=False,
                    )
                    torch.cuda.empty_cache()

            pred_adata = model.get_adata()
            try:
                mdir = (
                    model.logger.save_dir
                    if model.logger.save_dir is not None
                    else "data"
                )
            except:
                mdir = "data"
            file = (
                mdir
                + "/step_"
                + str(model.global_step)
                + "_"
                + model.name
                + "_predict_part_"
                + str(model.counter)
                + "_"
                + str(model.global_rank)
                + ".h5ad"
            )
            print("h5ad file path", file)

        if pred_adata is None:
            print(f"Loading from {file} instead of recomputing")
            pred_adata = sc.read_h5ad(file)
        else:
            print(f"Recomputed pred_adata")


        if self.output_expression == "all":
            adata.obsm["scprint_mu"] = model.expr_pred[0]
            adata.obsm["scprint_theta"] = model.expr_pred[1]
            adata.obsm["scprint_pi"] = model.expr_pred[2]
            adata.obsm["scprint_pos"] = model.pos.cpu().numpy()
        elif self.output_expression == "sample":
            adata.obsm["scprint_expr"] = (
                utils.zinb_sample(
                    model.expr_pred[0],
                    model.expr_pred[1],
                    model.expr_pred[2],
                )
                .cpu()
                .numpy()
            )
            adata.obsm["scprint_pos"] = model.pos.cpu().numpy()
        elif self.output_expression == "old":
            expr = np.array(model.expr_pred[0])
            expr[
                np.random.binomial(
                    1,
                    p=np.array(
                        torch.nn.functional.sigmoid(
                            model.expr_pred[2].to(torch.float32)
                        )
                    ),
                ).astype(bool)
            ] = 0
            expr[expr <= 0.3] = 0
            expr[(expr >= 0.3) & (expr <= 1)] = 1
            adata.obsm["scprint_expr"] = expr.astype(int)
            adata.obsm["scprint_pos"] = model.pos.cpu().numpy()

        pred_adata.obs.index = adata.obs.index
        try:
            adata.obsm["scprint_umap"] = pred_adata.obsm["X_umap"]
        except:
            print("too few cells to embed into a umap")
        try:
            adata.obsm["scprint_leiden"] = pred_adata.obsm["leiden"]
        except:
            print("too few cells to compute a clustering")
        adata.obsm["scprint"] = pred_adata.X
        pred_adata.obs.index = adata.obs.index
        adata.obs = pd.concat([adata.obs, pred_adata.obs], axis=1)
        if self.keep_all_cls_pred:
            allclspred = model.pred
            columns = []
            for cl in model.classes:
                n = model.label_counts[cl]
                columns += [model.label_decoders[cl][i] for i in range(n)]
            allclspred = pd.DataFrame(
                allclspred, columns=columns, index=adata.obs.index
            )
            adata.obs = pd.concat([adata.obs, allclspred], axis=1)

        metrics = {}
        if self.doclass and not self.keep_all_cls_pred:
            for cl in model.classes:
                res = []
                if cl not in adata.obs.columns:
                    continue
                class_topred = model.label_decoders[cl].values()

                if cl in model.labels_hierarchy:
                    cur_labels_hierarchy = {
                        model.label_decoders[cl][k]: [
                            model.label_decoders[cl][i] for i in v
                        ]
                        for k, v in model.labels_hierarchy[cl].items()
                    }
                else:
                    cur_labels_hierarchy = {}

                for pred, true in adata.obs[["pred_" + cl, cl]].values:
                    if pred == true:
                        res.append(True)
                        continue
                    if len(cur_labels_hierarchy) > 0:
                        if true in cur_labels_hierarchy:
                            res.append(pred in cur_labels_hierarchy[true])
                            continue
                        elif true not in class_topred:
                            raise ValueError(
                                f"true label {true} not in available classes"
                            )
                        elif true != "unknown":
                            res.append(False)
                    elif true not in class_topred:
                        raise ValueError(f"true label {true} not in available classes")
                    elif true != "unknown":
                        res.append(False)
                if len(res) == 0:
                    res = [1]
                if self.doplot:
                    print("    ", cl)
                    print("     accuracy:", sum(res) / len(res))
                    print(" ")
                metrics.update({cl + "_accuracy": sum(res) / len(res)})
        return adata, metrics


def compute_corr(
    out: np.ndarray,
    to: np.ndarray,
    doplot: bool = True,
    compute_mean_regress: bool = False,
    plot_corr_size: int = 64,
) -> dict:
    metrics = {}
    corr_coef, p_value = spearmanr(
        out,
        to.T,
    )
    corr_coef[p_value > 0.05] = 0

    val = plot_corr_size + 2 if compute_mean_regress else plot_corr_size
    metrics.update(
        {"recons_corr": np.mean(corr_coef[val:, :plot_corr_size].diagonal())}
    )
    if compute_mean_regress:
        metrics.update(
            {
                "mean_regress": np.mean(
                    corr_coef[
                        plot_corr_size : plot_corr_size + 2,
                        :plot_corr_size,
                    ].flatten()
                )
            }
        )
    if doplot:
        plt.figure(figsize=(10, 5))
        plt.imshow(corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Correlation Coefficient of expr and i["x"]')
        plt.show()
    return metrics


def default_benchmark(
    model: torch.nn.Module,
    default_dataset: str = "pancreas",
    do_class: bool = True,
    coarse: bool = False,
) -> dict:
    if default_dataset == "pancreas":
        adata = sc.read(
            FILE_LOC + "/../../data/pancreas_atlas.h5ad",
            backup_url="https://figshare.com/ndownloader/files/24539828",
        )
        adata.obs["cell_type_ontology_term_id"] = adata.obs["celltype"].replace(
            COARSE if coarse else FINE
        )
        adata.obs["assay_ontology_term_id"] = adata.obs["tech"].replace(
            COARSE if coarse else FINE
        )
    elif default_dataset == "lung":
        adata = sc.read(
            FILE_LOC + "/../../data/lung_atlas.h5ad",
            backup_url="https://figshare.com/ndownloader/files/24539942",
        )
        adata.obs["cell_type_ontology_term_id"] = adata.obs["cell_type"].replace(
            COARSE if coarse else FINE
        )
    else:
        adata = sc.read_h5ad(default_dataset)
        adata.obs["batch"] = adata.obs["assay_ontology_term_id"]
        adata.obs["cell_type"] = adata.obs["cell_type_ontology_term_id"]
    preprocessor = Preprocessor(
        use_layer="counts",
        is_symbol=True,
        force_preprocess=True,
        skip_validate=True,
        do_postp=False,
    )
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = preprocessor(adata.copy())
    embedder = Embedder(
        pred_embedding=["cell_type_ontology_term_id"],
        doclass=(default_dataset not in ["pancreas", "lung"]),
        max_len=4000,
        keep_all_cls_pred=False,
        output_expression="none",
    )
    embed_adata, metrics = embedder(model, adata.copy())

    bm = Benchmarker(
        embed_adata,
        batch_key="tech" if default_dataset == "pancreas" else "batch",
        label_key="celltype" if default_dataset == "pancreas" else "cell_type",
        embedding_obsm_keys=["scprint"],
        n_jobs=6,
    )
    bm.benchmark()
    metrics.update({"scib": bm.get_results(min_max_scale=False).T.to_dict()["scprint"]})
    metrics["classif"] = compute_classification(
        embed_adata, model.classes, model.label_decoders, model.labels_hierarchy
    )
    return metrics


def compute_classification(
    adata: AnnData,
    classes: List[str],
    label_decoders: Dict[str, Any],
    labels_hierarchy: Dict[str, Any],
    metric_type: List[str] = ["macro", "micro", "weighted"],
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for label in classes:
        res = []
        if label not in adata.obs.columns:
            continue
        labels_topred = label_decoders[label].values()
        if label in labels_hierarchy:
            parentdf = (
                bt.CellType.filter()
                .df(include=["parents__ontology_id"])
                .set_index("ontology_id")[["parents__ontology_id"]]
            )
            parentdf.parents__ontology_id = parentdf.parents__ontology_id.astype(str)
            class_groupings = {
                k: get_descendants(k, parentdf) for k in set(adata.obs[label].unique())
            }
        for pred, true in adata.obs[["pred_" + label, label]].values:
            if pred == true:
                res.append(true)
                continue
            if label in labels_hierarchy:
                if true in class_groupings:
                    res.append(true if pred in class_groupings[true] else "")
                    continue
                elif true not in labels_topred:
                    raise ValueError(f"true label {true} not in available classes")
            elif true not in labels_topred:
                raise ValueError(f"true label {true} not in available classes")
            res.append("")
        metrics[label] = {}
        metrics[label]["accuracy"] = np.mean(np.array(res) == adata.obs[label].values)
        for x in metric_type:
            metrics[label][x] = f1_score(
                np.array(res), adata.obs[label].values, average=x
            )
    return metrics


FINE = {
    "gamma": "CL:0002275",
    "beta": "CL:0000169",
    "epsilon": "CL:0005019",
    "acinar": "CL:0000622",
    "delta": "CL:0000173",
    "schwann": "CL:0002573",
    "activated_stellate": "CL:0000057",
    "alpha": "CL:0000171",
    "mast": "CL:0000097",
    "Mast cell": "CL:0000097",
    "quiescent_stellate": "CL:0000057",
    "t_cell": "CL:0000084",
    "endothelial": "CL:0000115",
    "Endothelium": "CL:0000115",
    "ductal": "CL:0002079",
    "macrophage": "CL:0000235",
    "Macrophage": "CL:0000235",
    "B cell": "CL:0000236",
    "Type 2": "CL:0002063",
    "Type 1": "CL:0002062",
    "Ciliated": "CL:4030034",
    "Dendritic cell": "CL:0000451",
    "Ionocytes": "CL:0005006",
    "Basal 1": "CL:0000646",
    "Basal 2": "CL:0000646",
    "Secretory": "CL:0000151",
    "Neutrophil_CD14_high": "CL:0000775",
    "Neutrophils_IL1R2": "CL:0000775",
    "Lymphatic": "CL:0002138",
    "Fibroblast": "CL:0000057",
    "T/NK cell": "CL:0000814",
    "inDrop1": "EFO:0008780",
    "inDrop3": "EFO:0008780",
    "inDrop4": "EFO:0008780",
    "inDrop2": "EFO:0008780",
    "fluidigmc1": "EFO:0010058",
    "smarter": "EFO:0010058",
    "celseq2": "EFO:0010010",
    "smartseq2": "EFO:0008931",
    "celseq": "EFO:0008679",
}
COARSE = {
    "beta": "CL:0008024",
    "epsilon": "CL:0008024",
    "delta": "CL:0008024",
    "alpha": "CL:0008024",
    "gamma": "CL:0008024",
    "acinar": "CL:0000150",
    "ductal": "CL:0000068",
    "schwann": "CL:0000125",
    "endothelial": "CL:0000115",
    "Endothelium": "CL:0000115",
    "Lymphatic": "CL:0000115",
    "macrophage": "CL:0000235",
    "Macrophage": "CL:0000235",
    "mast": "CL:0000097",
    "Mast cell": "CL:0000097",
    "Neutrophil_CD14_high": "CL:0000775",
    "Neutrophils_IL1R2": "CL:0000775",
    "t_cell": "CL:0000084",
    "T/NK cell": "CL:0000084",
    "B cell": "CL:0000236",
    "Dendritic cell": "CL:0000451",
    "activated_stellate": "CL:0000057",
    "quiescent_stellate": "CL:0000057",
    "Fibroblast": "CL:0000057",
    "Type 2": "CL:0000066",
    "Type 1": "CL:0000066",
    "Ionocytes": "CL:0000066",
    "Basal 1": "CL:0000066",
    "Basal 2": "CL:0000066",
    "Ciliated": "CL:0000064",
    "Secretory": "CL:0000151",
    "inDrop1": "EFO:0008780",
    "inDrop3": "EFO:0008780",
    "inDrop4": "EFO:0008780",
    "inDrop2": "EFO:0008780",
    "fluidigmc1": "EFO:0010058",
    "smarter": "EFO:0010058",
    "celseq2": "EFO:0010010",
    "smartseq2": "EFO:0008931",
    "celseq": "EFO:0008679",
}
