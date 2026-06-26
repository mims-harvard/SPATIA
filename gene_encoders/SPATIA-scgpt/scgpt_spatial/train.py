import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
import transformers
import tyro
from accelerate import Accelerator, DistributedDataParallelKwargs, GradScalerKwargs
from accelerate.utils import set_seed
from loguru import logger
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import wandb

from .configs import MainConfig
from .data_collator import DataCollator
from .dataset import SingleAdataDataset
from .dataset import MultiAdataDataset, create_dataloaders
from .loss import masked_mse_loss, masked_relative_error
from .model import TransformerModel
from .tasks.cell_emb import load_pretrained
from .tokenizer import GeneVocab


def setup_logging_and_saving(cfg: MainConfig) -> Tuple[Path, Optional[SummaryWriter]]:
    if cfg.trainer.exp_name is None:
        cfg.trainer.exp_name = f"scgpt-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{cfg.model.combine_weight}-{cfg.model.image_combine_weight}-{cfg.model.image_recon_loss_weight}-{random.randint(0, 9999)}"

    save_dir = Path(cfg.trainer.save_dir) / cfg.trainer.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="scgpt-spatial-train",
        name=cfg.trainer.exp_name,
        config=asdict(cfg),
    )

    with open(save_dir / "config.json", "w") as f:
        serializable_config = asdict(cfg)
        json.dump(serializable_config, f, indent=4)

    logger.add(save_dir / "run.log")
    logger.info(f"Configuration saved to {save_dir / 'config.json'}")

    writer = SummaryWriter(log_dir=save_dir / "tensorboard")
    return save_dir, writer


def get_num_types(dataset: MultiAdataDataset) -> int:
    logger.info("Calculating number of cell types...")
    all_cell_types = []
    num_samples_to_check = min(10000, len(dataset))
    for i in range(num_samples_to_check):
        item = dataset[i]
        if "celltype" in item:
            all_cell_types.append(item["celltype"].item())

    if not all_cell_types:
        logger.warning(
            "Could not determine number of cell types from dataset. Defaulting to 1."
        )
        return 1

    num_unique_types = len(set(all_cell_types))
    logger.info(f"Found {num_unique_types} unique cell types.")
    return num_unique_types


def train_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    writer: SummaryWriter,
    save_dir: Path,
    mask_value: int,
    cfg: MainConfig,
    vocab: GeneVocab,
    val_loader: DataLoader,
    best_val_loss: float,
    num_types: int,
) -> None:
    model.train()
    total_loss, total_mse, total_cls, total_mvc = 0.0, 0.0, 0.0, 0.0
    total_mre = 0.0
    total_recon = 0.0
    total_cls_acc = 0.0
    start_time = time.time()

    for step, data_dict in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        dynamic_ncols=True,
    ):
        if accelerator.is_main_process and step == 0:
            logger.info("Starting first training step...")

        global_step = epoch * len(train_loader) + step

        with torch.cuda.amp.autocast(enabled=cfg.trainer.fp16):
            if accelerator.is_main_process and step == 0:
                logger.info("Starting model forward pass...")

            use_generative = cfg.trainer.training_tasks in ["gen", "both"]
            use_cls = not cfg.model.no_cls
            use_mvc = True

            if use_generative:
                input_dict = {
                    "pcpt_genes": data_dict["pcpt_gene"],
                    "pcpt_values": data_dict["pcpt_expr"],
                    "gen_genes": data_dict["gen_gene"],
                    "pcpt_key_padding_mask": data_dict["pcpt_key_padding_mask"],
                    "gen_key_padding_mask": data_dict["gen_key_padding_mask"],
                    "image": data_dict["image"],
                }
            else:
                raise NotImplementedError(
                    "Perceptual training mode is not implemented in this script."
                )
            output_dict = model(
                **input_dict,
                generative_training=use_generative,
                CLS=use_cls,
                MVC=use_mvc,
            )

            if accelerator.is_main_process and step == 0:
                logger.info("Model forward pass completed, calculating losses...")

            loss = 0.0
            loss_mse, loss_cls, loss_mvc_val = 0.0, 0.0, 0.0
            loss_recon = 0.0
            cls_acc = 0.0

            if cfg.model.image_encoder_cls is not None:
                loss_recon = output_dict["recon_loss"]

            if use_generative:
                if "pcpt_expr_target" in data_dict:
                    positions_to_match = data_dict["pcpt_expr"].eq(mask_value)
                    loss_mse += masked_mse_loss(
                        output_dict["pcpt_preds"],
                        data_dict["pcpt_expr_target"],
                        positions_to_match,
                    )
                positions_to_match = ~data_dict["gen_key_padding_mask"]
                loss_mse += masked_mse_loss(
                    output_dict["gen_preds"],
                    data_dict["gen_expr_target"],
                    positions_to_match,
                )
                if use_cls and num_types > 1:
                    target_labels = data_dict["cell_type"]
                    loss_cls = nn.CrossEntropyLoss()(
                        output_dict["cls_output"], target_labels
                    )
                    with torch.no_grad():
                        predictions = torch.argmax(output_dict["cls_output"], dim=1)
                        cls_acc = (predictions == target_labels).float().mean().item()
                if use_mvc:
                    loss_mvc_val = masked_mse_loss(
                        output_dict["mvc_output"][:, data_dict["pcpt_gene"].shape[1] :],
                        data_dict["gen_expr_target"],
                        positions_to_match,
                    )
            else:
                positions_to_match = data_dict["masked_expr"].eq(mask_value)
                loss_mse = masked_mse_loss(
                    output_dict["mlm_output"], data_dict["expr"], positions_to_match
                )
                if use_cls and num_types > 1:
                    target_labels = data_dict[
                        "cell_type"
                    ]
                    loss_cls = nn.CrossEntropyLoss()(
                        output_dict["cls_output"], target_labels
                    )
                    with torch.no_grad():
                        predictions = torch.argmax(output_dict["cls_output"], dim=1)
                        cls_acc = (predictions == target_labels).float().mean().item()
                if use_mvc:
                    loss_mvc_val = masked_mse_loss(
                        output_dict["mvc_output"], data_dict["expr"], positions_to_match
                    )

            loss += loss_mse
            if loss_cls > 0:
                loss += loss_cls
            if loss_mvc_val > 0:
                loss += loss_mvc_val
            if loss_recon > 0:
                loss += loss_recon

            loss = loss / cfg.trainer.grad_accu_steps
            accelerator.backward(loss)

        if (step + 1) % cfg.trainer.grad_accu_steps == 0 or step == len(
            train_loader
        ) - 1:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            if use_generative:
                mre = masked_relative_error(
                    output_dict["gen_preds"],
                    data_dict["gen_expr_target"],
                    positions_to_match,
                )
            else:
                mre = masked_relative_error(
                    output_dict["mlm_output"], data_dict["expr"], positions_to_match
                )

        if accelerator.is_main_process and step == 0:
            logger.info("Starting metrics gathering...")

        avg_loss = (
            accelerator.gather(loss.repeat(cfg.trainer.batch_size)).mean().item()
            * cfg.trainer.grad_accu_steps
        )
        avg_mse = (
            accelerator.gather(loss_mse.repeat(cfg.trainer.batch_size)).mean().item()
        )
        avg_mre = accelerator.gather(mre.repeat(cfg.trainer.batch_size)).mean().item()
        loss_cls_tensor = torch.tensor(
            loss_cls if (use_cls and loss_cls > 0) else 0.0, device=accelerator.device
        )
        avg_cls = (
            accelerator.gather(loss_cls_tensor.repeat(cfg.trainer.batch_size))
            .mean()
            .item()
        )

        loss_mvc_tensor = torch.tensor(
            loss_mvc_val if (use_mvc and loss_mvc_val > 0) else 0.0,
            device=accelerator.device,
        )
        avg_mvc = (
            accelerator.gather(loss_mvc_tensor.repeat(cfg.trainer.batch_size))
            .mean()
            .item()
        )

        loss_recon_tensor = torch.tensor(
            loss_recon if loss_recon > 0 else 0.0, device=accelerator.device
        )
        avg_recon = (
            accelerator.gather(loss_recon_tensor.repeat(cfg.trainer.batch_size))
            .mean()
            .item()
        )
        cls_acc_tensor = torch.tensor(
            cls_acc if (use_cls and num_types > 1) else 0.0, device=accelerator.device
        )
        avg_cls_acc = (
            accelerator.gather(cls_acc_tensor.repeat(cfg.trainer.batch_size))
            .mean()
            .item()
        )

        if accelerator.is_main_process and step == 0:
            logger.info("Metrics gathering completed, updating totals...")

        total_loss += avg_loss
        total_mse += avg_mse
        total_mre += avg_mre
        total_cls += avg_cls
        total_mvc += avg_mvc
        total_recon += avg_recon
        total_cls_acc += avg_cls_acc

        if accelerator.is_main_process and (step + 1) % cfg.trainer.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time

            log_str = (
                f"| epoch {epoch:3d} | step {step+1:5d}/{len(train_loader):5d} | "
                f"lr {lr:02.6f} | ms/batch {elapsed * 1000 / cfg.trainer.log_interval:5.2f} | "
                f"loss {total_loss/cfg.trainer.log_interval:5.4f} | "
                f"mse {total_mse/cfg.trainer.log_interval:5.4f} | mre {total_mre/cfg.trainer.log_interval:5.4f}"
                f" | recon {total_recon/cfg.trainer.log_interval:5.4f}"
            )
            if use_cls and num_types > 1 and total_cls > 0:
                log_str += f" | cls {total_cls/cfg.trainer.log_interval:5.4f}"
                if use_cls and num_types > 1 and total_cls_acc > 0:
                    log_str += (
                        f" | cls_acc {total_cls_acc/cfg.trainer.log_interval:5.4f}"
                    )
            if use_mvc and total_mvc > 0:
                log_str += f" | mvc {total_mvc/cfg.trainer.log_interval:5.4f}"
            accelerator.print(log_str)

            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/mse", avg_mse, global_step)
            writer.add_scalar("train/mre", avg_mre, global_step)
            if use_cls and num_types > 1 and avg_cls > 0:
                writer.add_scalar("train/cls_loss", avg_cls, global_step)
                if use_cls and num_types > 1 and avg_cls_acc > 0:
                    writer.add_scalar("train/cls_acc", avg_cls_acc, global_step)
            if use_mvc and avg_mvc > 0:
                writer.add_scalar("train/mvc_loss", avg_mvc, global_step)
            if avg_recon > 0:
                writer.add_scalar("train/recon_loss", avg_recon, global_step)
            writer.add_scalar("lr", lr, global_step)

            log_dict = {
                "train/loss": avg_loss,
                "train/mse": avg_mse,
                "train/mre": avg_mre,
                "lr": lr,
            }
            if use_cls and num_types > 1 and avg_cls > 0:
                log_dict["train/cls_loss"] = avg_cls
                if use_cls and num_types > 1 and avg_cls_acc > 0:
                    log_dict["train/cls_acc"] = avg_cls_acc
            if use_mvc and avg_mvc > 0:
                log_dict["train/mvc_loss"] = avg_mvc
            if avg_recon > 0:
                log_dict["train/recon_loss"] = avg_recon
            wandb.log(log_dict, step=global_step)

            (
                total_loss,
                total_mse,
                total_mre,
                total_cls,
                total_mvc,
                total_recon,
                total_cls_acc,
            ) = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            start_time = time.time()

        if (global_step + 1) % cfg.trainer.eval_interval_steps == 0:
            best_val_loss = evaluate_and_log(
                model=model,
                val_loader=val_loader,
                accelerator=accelerator,
                cfg=cfg,
                vocab=vocab,
                epoch=epoch,
                global_step=global_step,
                writer=writer,
                save_dir=save_dir,
                best_val_loss=best_val_loss,
                num_types=num_types,
            )
            model.train()

        if (
            global_step + 1
        ) % cfg.trainer.save_interval_steps == 0 and accelerator.is_main_process:
            accelerator.print(f"Saving checkpoint at global step {global_step+1}")
            accelerator.save_state(
                output_dir=str(save_dir / f"checkpoint-step-{global_step+1}")
            )
    return best_val_loss


def generate_cell_embeddings(
    model: nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    vocab: GeneVocab,
    pad_token: str,
    fp16: bool,
) -> np.ndarray:
    all_embeddings = []

    unwrapped_model = accelerator.unwrap_model(model)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16):
        for batch in tqdm(
            loader, desc="Generating test embeddings", dynamic_ncols=True
        ):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(accelerator.device)

            input_dict = {
                "src": batch["gene"],
                "values": batch["expr"],
                "src_key_padding_mask": batch["gene"].eq(vocab[pad_token]),
                "image": batch["image"],
            }

            outputs = unwrapped_model._encode_spatial(**input_dict)

            embeddings = outputs["expression_latent"][:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

    full_embeddings = np.concatenate(all_embeddings, axis=0)
    normalized_embeddings = full_embeddings / np.linalg.norm(
        full_embeddings, axis=1, keepdims=True
    )
    return normalized_embeddings


def perform_clustering(
    data: np.ndarray,
    method="kmeans",
    n_clusters: int = None,
    resolution: float = 0.5,
    random_state: int = 42,
) -> np.ndarray:
    logger.info(f"Performing {method} clustering with {n_clusters} clusters")
    if method.lower() == "kmeans":
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=20
        ).fit(data)
        return kmeans.labels_
    elif method.lower() == "leiden":
        adata = sc.AnnData(data)
        sc.pp.neighbors(adata, use_rep="X", random_state=random_state)
        sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
        return np.array(adata.obs["leiden"].astype(int))
    elif method.lower() == "louvain":
        adata = sc.AnnData(data)
        sc.pp.neighbors(adata, use_rep="X", random_state=random_state)
        sc.tl.louvain(adata, resolution=resolution, random_state=random_state)
        return np.array(adata.obs["louvain"].astype(int))
    else:
        raise ValueError(f"Unsupported clustering method: {method}")


def visualize_clustering_results(
    adata,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_key: str,
    embeddings: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    coords = None
    coord_labels = None

    if embeddings.shape[1] > 2:
        try:
            import umap

            logger.info(
                "Applying UMAP to reduce embedding dimensions for visualization"
            )
            reducer = umap.UMAP(
                n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
            )
            coords = reducer.fit_transform(embeddings)
            coord_labels = ["UMAP 1", "UMAP 2"]
        except ImportError:
            logger.warning("UMAP not available, using PCA for dimensionality reduction")
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(embeddings)
            coord_labels = [
                f"PC 1 ({pca.explained_variance_ratio_[0]:.2%})",
                f"PC 2 ({pca.explained_variance_ratio_[1]:.2%})",
            ]
    else:
        coords = embeddings
        coord_labels = ["Embedding Dim 1", "Embedding Dim 2"]

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 8)
    )

    unique_true_labels = np.unique(true_labels)
    n_true_labels = len(unique_true_labels)

    if n_true_labels <= 10:
        colors_true = plt.cm.tab10(np.linspace(0, 1, n_true_labels))
    elif n_true_labels <= 20:
        colors_true = plt.cm.tab20(np.linspace(0, 1, n_true_labels))
    else:
        colors_true = plt.cm.viridis(np.linspace(0, 1, n_true_labels))

    for i, label in enumerate(unique_true_labels):
        mask = true_labels == label
        ax1.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors_true[i]],
            label=label,
            s=12,
            alpha=0.8,
            edgecolors="none",
            rasterized=True,
        )

    ax1.set_title(f"Ground Truth ({label_key})", fontsize=16, fontweight="bold", pad=20)
    ax1.set_xlabel(coord_labels[0], fontsize=12)
    ax1.set_ylabel(coord_labels[1], fontsize=12)
    ax1.tick_params(axis="both", which="major", labelsize=10)

    if n_true_labels <= 10:
        ax1.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
    elif n_true_labels <= 20:
        ax1.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
            ncol=1,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
    else:
        ncol = min(3, max(1, n_true_labels // 15))
        ax1.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=6,
            ncol=ncol,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.set_axisbelow(True)

    unique_pred_labels = np.unique(pred_labels)
    n_pred_labels = len(unique_pred_labels)

    if n_pred_labels <= 10:
        colors_pred = plt.cm.tab10(np.linspace(0, 1, n_pred_labels))
    elif n_pred_labels <= 20:
        colors_pred = plt.cm.tab20(np.linspace(0, 1, n_pred_labels))
    else:
        colors_pred = plt.cm.viridis(np.linspace(0, 1, n_pred_labels))

    for i, label in enumerate(unique_pred_labels):
        mask = pred_labels == label
        ax2.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors_pred[i]],
            label=f"Cluster {label}",
            s=12,
            alpha=0.8,
            edgecolors="none",
            rasterized=True,
        )

    ax2.set_title(
        "Predicted Clustering (Model Embeddings)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xlabel(coord_labels[0], fontsize=12)
    ax2.set_ylabel(coord_labels[1], fontsize=12)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    if n_pred_labels <= 10:
        ax2.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
    elif n_pred_labels <= 20:
        ax2.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
            ncol=1,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
    else:
        ncol = min(3, max(1, n_pred_labels // 15))
        ax2.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=6,
            ncol=ncol,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    if save_path:
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        logger.info(f"Clustering visualization saved to {save_path}")

    return fig


def evaluate_clustering(
    model: nn.Module,
    accelerator: Accelerator,
    cfg: MainConfig,
    vocab: GeneVocab,
) -> Dict[str, float]:
    if not cfg.data.test_h5ad_path or not Path(cfg.data.test_h5ad_path).exists():
        logger.warning(
            f"Test h5ad file not found at {cfg.data.test_h5ad_path}, skipping clustering eval."
        )
        return {}

    logger.info(f"Starting clustering evaluation on {cfg.data.test_h5ad_path}")

    adata = sc.read_h5ad(cfg.data.test_h5ad_path)
    label_key = "cell_type"
    if "cell_type" in adata.obs.columns:
        pass
    elif "celltype" in adata.obs.columns:
        label_key = "celltype"
    elif "annotation" in adata.obs.columns:
        label_key = "annotation"

    if label_key not in adata.obs.columns:
        logger.error(
            f"Could not find ground truth labels in obs columns ('cell_type', 'celltype', 'annotation'). Skipping clustering eval."
        )
        return {}
    logger.info(f"Using '{label_key}' as ground truth labels for clustering.")

    true_labels = adata.obs[label_key].values
    label_encoder = LabelEncoder()
    true_labels_encoded = label_encoder.fit_transform(true_labels)

    test_dataset = SingleAdataDataset(
        adata_path=cfg.data.test_h5ad_path,
        vocab=vocab,
        gene_stats_file=Path(cfg.data.gene_stats_file),
        gene_col=cfg.data.gene_col,
        cls_token="<cls>",
        pad_value=0.0,
        require_log1p=True,
        spatial_datadir=cfg.data.spatial_datadir,
        fix_missing_images=cfg.data.fix_missing_images,
        preprocessor_cls=cfg.data.preprocessor_cls,
        inference=True,
    )

    if cfg.data.input_emb_style == "category":
        pad_value = cfg.data.n_bins
    else:
        pad_value = -2

    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[cfg.trainer.pad_token],
        pad_value=pad_value,
        do_mlm=False,
        do_binning=(cfg.data.input_style == "binned"),
        n_bins=cfg.data.n_bins,
        max_length=cfg.data.max_seq_len,
        sampling=cfg.data.trunc_by_sample,
        data_style="pcpt",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )
    assert (
        accelerator.is_main_process
    ), "Clustering evaluation should only run on the main process."

    embeddings = generate_cell_embeddings(
        model, test_loader, accelerator, vocab, cfg.trainer.pad_token, cfg.trainer.fp16
    )

    pred_labels = perform_clustering(
        embeddings,
        method=cfg.trainer.clustering_method,
        n_clusters=adata.obs[label_key].nunique(),
        resolution=cfg.trainer.clustering_resolution,
    )

    ari_score = adjusted_rand_score(true_labels_encoded, pred_labels)
    nmi_score = normalized_mutual_info_score(true_labels_encoded, pred_labels)

    logger.info(f"Clustering results: ARI={ari_score:.4f}, NMI={nmi_score:.4f}")

    try:
        logger.info("Generating clustering visualization...")
        fig = visualize_clustering_results(
            adata=adata,
            true_labels=true_labels,
            pred_labels=pred_labels,
            label_key=label_key,
            embeddings=embeddings,
        )

        wandb.log({"test/clustering_result": wandb.Image(fig)})
        logger.info("Clustering visualization logged to wandb")

        plt.close(fig)

    except Exception as e:
        logger.warning(f"Failed to generate clustering visualization: {e}")

    return {"test/ARI": ari_score, "test/NMI": nmi_score}


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
    cfg: MainConfig,
    vocab: GeneVocab,
    num_types: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mre = 0.0
    total_recon = 0.0
    total_cls_acc = 0.0
    num_cls_samples = 0

    for data_dict in val_loader:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.trainer.fp16):
            use_generative = cfg.trainer.training_tasks in ["gen", "both"]
            use_cls = not cfg.model.no_cls and num_types > 1
            if use_generative:
                input_dict = {
                    "pcpt_genes": data_dict["pcpt_gene"],
                    "pcpt_values": data_dict["pcpt_expr"],
                    "gen_genes": data_dict["gen_gene"],
                    "pcpt_key_padding_mask": data_dict["pcpt_key_padding_mask"],
                    "gen_key_padding_mask": data_dict["gen_key_padding_mask"],
                    "image": data_dict["image"],
                }
            else:
                raise NotImplementedError(
                    "Perceptual training mode is not implemented in this script."
                )

            output_dict = model(
                **input_dict, generative_training=use_generative, CLS=use_cls, MVC=False
            )

            if use_generative:
                positions_to_match = ~data_dict["gen_key_padding_mask"]
                loss = masked_mse_loss(
                    output_dict["gen_preds"],
                    data_dict["gen_expr_target"],
                    positions_to_match,
                )
                mre = masked_relative_error(
                    output_dict["gen_preds"],
                    data_dict["gen_expr_target"],
                    positions_to_match,
                )
            else:
                positions_to_match = data_dict["masked_expr"].eq(cfg.trainer.mask_value)
                loss = masked_mse_loss(
                    output_dict["mlm_output"], data_dict["expr"], positions_to_match
                )
                mre = masked_relative_error(
                    output_dict["mlm_output"], data_dict["expr"], positions_to_match
                )

            if use_cls and "cls_output" in output_dict:
                target_labels = data_dict["cell_type"]
                predictions = torch.argmax(output_dict["cls_output"], dim=1)
                cls_acc = (predictions == target_labels).float().mean().item()
                total_cls_acc += cls_acc
                num_cls_samples += 1

        total_loss += loss.item()
        total_mre += mre.item()
        if cfg.model.image_encoder_cls is not None:
            total_recon += output_dict["recon_loss"].item()

    all_losses = accelerator.gather(torch.tensor(total_loss, device=accelerator.device))
    all_mres = accelerator.gather(torch.tensor(total_mre, device=accelerator.device))
    all_recons = accelerator.gather(
        torch.tensor(total_recon, device=accelerator.device)
    )
    all_cls_accs = accelerator.gather(
        torch.tensor(total_cls_acc, device=accelerator.device)
    )
    all_cls_samples = accelerator.gather(
        torch.tensor(num_cls_samples, device=accelerator.device)
    )

    num_batches = torch.tensor(len(val_loader), device=accelerator.device)
    total_batches = accelerator.gather(num_batches).sum().item()

    avg_loss = all_losses.sum().item() / total_batches
    avg_mre = all_mres.sum().item() / total_batches
    avg_recon = all_recons.sum().item() / total_batches

    total_cls_samples = all_cls_samples.sum().item()
    avg_cls_acc = (
        all_cls_accs.sum().item() / total_cls_samples if total_cls_samples > 0 else 0.0
    )

    metrics = {"val_loss_mse": avg_loss, "val_mre": avg_mre, "val_recon": avg_recon}

    if total_cls_samples > 0:
        metrics["val_cls_acc"] = avg_cls_acc

    model.eval()
    if cfg.data.test_h5ad_path and accelerator.is_main_process:
        accelerator.print("Running clustering evaluation on test set...")
        clustering_metrics = evaluate_clustering(model, accelerator, cfg, vocab)
        metrics.update(clustering_metrics)
    model.train()

    return metrics


def evaluate_and_log(
    model: nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
    cfg: MainConfig,
    vocab: GeneVocab,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    save_dir: Path,
    best_val_loss: float,
    num_types: int,
) -> float:
    accelerator.print(f"--- Starting evaluation at global step {global_step} ---")
    val_metrics = evaluate(model, val_loader, accelerator, cfg, vocab, num_types)

    if not accelerator.is_main_process:
        return best_val_loss

    accelerator.print("Evaluation complete.")

    val_loss = val_metrics["val_loss_mse"]
    val_mre = val_metrics["val_mre"]
    val_recon = val_metrics["val_recon"]
    val_cls_acc = val_metrics.get("val_cls_acc", None)

    log_line = (
        f"| epoch {epoch:3d} | step {global_step:8d} | "
        f"valid loss/mse {val_loss:5.4f} | valid mre {val_mre:5.4f} | valid recon {val_recon:5.4f}"
    )
    if val_cls_acc is not None:
        log_line += f" | valid cls_acc {val_cls_acc:.4f}"
    if "test/ARI" in val_metrics:
        log_line += f" | test ARI {val_metrics['test/ARI']:.4f}"
    if "test/NMI" in val_metrics:
        log_line += f" | test NMI {val_metrics['test/NMI']:.4f}"

    accelerator.print("-" * 89)
    accelerator.print(log_line)
    accelerator.print("-" * 89)

    log_dict = {
        "valid/mse": val_loss,
        "valid/mre": val_mre,
        "valid/recon": val_recon,
    }
    if val_cls_acc is not None:
        log_dict["valid/cls_acc"] = val_cls_acc
    if "test/ARI" in val_metrics:
        log_dict["test/ARI"] = val_metrics["test/ARI"]
    if "test/NMI" in val_metrics:
        log_dict["test/NMI"] = val_metrics["test/NMI"]

    wandb.log(log_dict, step=global_step)
    for key, value in log_dict.items():
        writer.add_scalar(key, value, global_step)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        accelerator.print(
            f"New best validation loss: {best_val_loss:.4f}. Saving model..."
        )
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), save_dir / "best_model.pt")
        accelerator.save(unwrapped_model, save_dir / "best_model_full.pt")

    return best_val_loss


def main(cfg: MainConfig) -> None:
    grad_scaler_kwargs = {"enabled": cfg.trainer.fp16}
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.grad_accu_steps,
        kwargs_handlers=[GradScalerKwargs(**grad_scaler_kwargs), ddp_kwargs],
    )

    if accelerator.is_main_process:
        save_dir, writer = setup_logging_and_saving(cfg)
    else:
        save_dir, writer = None, None

    vocab = GeneVocab.from_file(Path(cfg.data.vocab_path))
    special_tokens = [cfg.trainer.pad_token, "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
            accelerator.print(f"Added special token '{s}' to vocabulary.")

    if accelerator.is_main_process:
        with open(save_dir / "vocab.json", "w") as f:
            json.dump(vocab.get_stoi(), f, indent=2)

    if cfg.model.load_model_path:
        args_json_path = Path(cfg.model.load_model_path) / "args.json"
        if args_json_path.exists():
            accelerator.print(f"Loading model args from {args_json_path}")
            with open(args_json_path, "r") as f:
                loaded_args = json.load(f)

            updated_params = []
            for key, value in loaded_args.items():
                target_key = key
                if key == "fast_transformer":
                    target_key = "use_fast_transformer"

                if hasattr(cfg.model, target_key):
                    current_value = getattr(cfg.model, target_key)
                    setattr(cfg.model, target_key, value)
                    updated_params.append(
                        f"model.{target_key} (from args.json key '{key}') {current_value} -> {value}"
                    )

            if updated_params and accelerator.is_main_process:
                accelerator.print(f"Updated model config from {args_json_path}:")
                for param_update in updated_params:
                    accelerator.print(f"  - {param_update}")
        else:
            if accelerator.is_main_process:
                accelerator.print(
                    f"Warning: args.json not found at {args_json_path}. Using model config from command line or defaults."
                )

    accelerator.print("Loading data...")
    data_dir = Path(cfg.data.data_source)
    if not data_dir.is_dir():
        raise ValueError(
            f"data_source '{cfg.data.data_source}' must be a directory of .h5ad files."
        )

    adata_paths = list(data_dir.glob("*.h5ad"))
    if not adata_paths:
        raise ValueError(f"No .h5ad files found in '{data_dir}'.")

    full_dataset = MultiAdataDataset(
        adata_paths=adata_paths,
        vocab=vocab,
        gene_stats_file=Path(cfg.data.gene_stats_file),
        gene_col=cfg.data.gene_col,
        cls_token="<cls>",
        pad_value=0.0,
        require_log1p=True,
        spatial_datadir=cfg.data.spatial_datadir,
        fix_missing_images=cfg.data.fix_missing_images,
        preprocessor_cls=cfg.data.preprocessor_cls,
        cell_type_col=cfg.data.cell_type_col,
    )

    if cfg.data.input_emb_style == "category":
        mask_value = cfg.data.n_bins + 1
        pad_value = cfg.data.n_bins
    else:
        mask_value = -1
        pad_value = -2

    if cfg.trainer.training_tasks in ["gen", "both"]:
        if isinstance(cfg.trainer.mask_ratio, float):
            mask_ratio = [0.25, 0.50, 0.75]
            logger.warning(
                "Using multiple mask ratios for generative training: "
                f"{mask_ratio} instead of <default_mask_ratio>."
            )
        else:
            mask_ratio = cfg.trainer.mask_ratio
    else:
        mask_ratio = cfg.trainer.mask_ratio

    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[cfg.trainer.pad_token],
        pad_value=pad_value,
        do_mlm=True,
        do_binning=(cfg.data.input_style == "binned"),
        n_bins=cfg.data.n_bins,
        mlm_probability=mask_ratio,
        mask_value=mask_value,
        max_length=cfg.data.max_seq_len,
        sampling=cfg.data.trunc_by_sample,
        data_style=cfg.trainer.training_tasks,
        cell_types=full_dataset.cell_types,
    )

    train_loader, val_loader = create_dataloaders(
        full_dataset,
        data_collator=collator,
        batch_size=cfg.trainer.batch_size,
        validation_split=cfg.data.valid_size_or_ratio,
        num_workers=4,
    )
    accelerator.print(
        f"Data loading complete. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    n_input_bins = (
        (cfg.data.n_bins + 2)
        if cfg.data.input_emb_style == "category"
        else cfg.data.n_bins
    )

    num_types = full_dataset.num_cell_types
    logger.info(f"Number of cell types: {num_types}")
    logger.info(f"Cell types: {full_dataset.cell_types}")
    if num_types > 1:
        logger.info(f"Using {num_types} cell types for classification head.")
        cfg.model.no_cls = False

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg.model.embsize,
        nhead=cfg.model.nheads,
        d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers,
        nlayers_cls=cfg.model.n_layers_cls,
        n_cls=num_types,
        vocab=vocab,
        dropout=cfg.model.dropout,
        pad_token=cfg.trainer.pad_token,
        pad_value=pad_value,
        do_mvc=True,
        do_dab=False,
        use_generative_training=(cfg.trainer.training_tasks in ["gen", "both"]),
        use_batch_labels=False,
        explicit_zero_prob=False,
        use_fast_transformer=cfg.model.use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
        use_MVC_impute=True,
        use_moe_dec=True,
        n_input_bins=n_input_bins,
        input_emb_style=cfg.data.input_emb_style,
        image_encoder_cls=cfg.model.image_encoder_cls,
        combine_weight=cfg.model.combine_weight,
        image_combine_weight=cfg.model.image_combine_weight,
        image_recon_loss_weight=cfg.model.image_recon_loss_weight,
    )

    if cfg.model.load_model_path:
        accelerator.print(f"Loading model from {cfg.model.load_model_path}")
        load_pretrained(
            model,
            torch.load(
                Path(cfg.model.load_model_path) / "best_model.pt",
                map_location="cpu",
            ),
            verbose=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
    )

    num_training_steps = len(train_loader) * cfg.trainer.epochs

    if cfg.optim.warmup_ratio_or_step > 0:
        warmup_steps = (
            int(num_training_steps * cfg.optim.warmup_ratio_or_step)
            if cfg.optim.warmup_ratio_or_step < 1
            else int(cfg.optim.warmup_ratio_or_step)
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, cfg.optim.scheduler_interval, gamma=cfg.optim.scheduler_factor
        )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_val_loss = float("inf")
    accelerator.print("Starting training...")

    best_val_loss = evaluate_and_log(
        model=model,
        val_loader=val_loader,
        accelerator=accelerator,
        cfg=cfg,
        vocab=vocab,
        epoch=0,
        global_step=0,
        writer=writer,
        save_dir=save_dir,
        best_val_loss=best_val_loss,
        num_types=num_types,
    )
    model.train()
    for epoch in range(0, cfg.trainer.epochs):
        best_val_loss = train_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            scheduler,
            accelerator,
            writer,
            save_dir,
            mask_value,
            cfg,
            vocab,
            val_loader,
            best_val_loss,
            num_types,
        )

    if accelerator.is_main_process:
        writer.close()
        wandb.finish()
    accelerator.print("Training finished.")


if __name__ == "__main__":
    cfg = tyro.cli(MainConfig)
    main(cfg)
