
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import xgboost as xgb
from PIL import Image
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from tqdm import tqdm

NSDIR = os.environ.get("SPATIA_BASELINES_DIR", "./baselines")
REPO = Path(__file__).resolve().parents[2]
SCGPT_PATH = REPO / "gene_encoders" / "SPATIA-scgpt"
sys.path.insert(0, str(SCGPT_PATH))


def parse_args():
    p = argparse.ArgumentParser(description="SPATIA image encoder on HEST-Bench (Table 3)")
    p.add_argument("--checkpoint-dir", default=os.environ.get(
        "SPATIA_CHECKPOINT_DIR", "./checkpoints"))
    p.add_argument("--bench-data-root", default=f"{NSDIR}/hest/bench_data")
    p.add_argument("--embed-root", default=f"{NSDIR}/hest/embeddings")
    p.add_argument("--results-dir", default=f"{NSDIR}/hest/results")
    p.add_argument("--datasets", nargs="+", default=["IDC"],
                   help="HEST-Bench datasets, e.g. IDC PAAD SKCM COAD LUAD")
    p.add_argument("--download", action="store_true",
                   help="Download the selected datasets from HuggingFace first")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--method", default="xgboost", choices=["xgboost", "ridge"])
    p.add_argument("--dimreduce", default="PCA", choices=["PCA", "none"])
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable log1p normalization of expression targets")
    return p.parse_args()


class SpatiaImageEncoder(nn.Module):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = self.checkpoint_dir / "config.json"
        self.weight_path = self.checkpoint_dir / "best_model.pt"
        self.vocab_path = self.checkpoint_dir / "vocab.json"
        self.model = None
        self.vocab = None
        self.cfg = None
        self.embed_dim = 768

    def load_model(self):
        from scgpt_spatial.model import TransformerModel
        from scgpt_spatial.tasks.cell_emb import load_pretrained
        from scgpt_spatial.tokenizer import GeneVocab

        with open(self.config_path) as f:
            self.cfg = json.load(f)
        model_cfg = self.cfg["model"]
        data_cfg = self.cfg["data"]
        trainer_cfg = self.cfg["trainer"]

        self.vocab = GeneVocab.from_file(self.vocab_path)
        for token in [trainer_cfg["pad_token"], "<cls>", "<eoc>"]:
            if token not in self.vocab:
                self.vocab.append_token(token)

        pad_value = -2
        n_input_bins = data_cfg["n_bins"]

        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=model_cfg["embsize"],
            nhead=model_cfg["nheads"],
            d_hid=model_cfg["d_hid"],
            nlayers=model_cfg["nlayers"],
            nlayers_cls=model_cfg["n_layers_cls"],
            n_cls=1,
            vocab=self.vocab,
            dropout=model_cfg["dropout"],
            pad_token=trainer_cfg["pad_token"],
            pad_value=pad_value,
            do_mvc=True,
            do_dab=False,
            use_generative_training=False,
            use_batch_labels=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
            use_MVC_impute=True,
            use_moe_dec=True,
            n_input_bins=n_input_bins,
            input_emb_style=data_cfg["input_emb_style"],
            image_encoder_cls=model_cfg["image_encoder_cls"],
            combine_weight=model_cfg["combine_weight"],
            image_combine_weight=model_cfg["image_combine_weight"],
            image_recon_loss_weight=model_cfg["image_recon_loss_weight"],
        )

        state_dict = torch.load(self.weight_path, map_location="cpu")
        load_pretrained(self.model, state_dict, verbose=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded SPATIA: gene {model_cfg['embsize']}-d, "
              f"image {model_cfg['image_encoder_cls']} {self.embed_dim}-d")
        return self

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images):
        if self.model is None:
            self.load_model()
        image_outputs = self.model._get_image_outputs(images)
        return image_outputs.last_hidden_state[:, 0, :]

    def extract_embeddings_from_h5(self, h5_path, batch_size=64):
        if self.model is None:
            self.load_model()
        transform = self.get_transforms()
        with h5py.File(h5_path, "r") as f:
            imgs = f["img"][:]

        all_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(imgs), batch_size), desc="extract"):
                batch_imgs = imgs[i:i + batch_size]
                batch = torch.stack(
                    [transform(Image.fromarray(im)) for im in batch_imgs]
                ).to(self.device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    emb = self.forward(batch)
                all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)


def download_datasets(datasets, bench_data_root):
    from huggingface_hub import snapshot_download

    for dataset in datasets:
        print(f"downloading {dataset} ...")
        snapshot_download(
            repo_id="MahmoodLab/hest-bench",
            repo_type="dataset",
            local_dir=bench_data_root,
            allow_patterns=[f"{dataset}/*", f"{dataset}/**/*"],
            ignore_patterns=["fm_v1/*"],
        )


def train_test_regression(X_train, X_test, y_train, y_test, genes, method, random_state):
    if method == "xgboost":
        reg = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1,
            subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1,
            random_state=random_state, n_jobs=-1,
        )
        reg.fit(X_train, y_train)
        preds_all = reg.predict(X_test)
    else:
        from sklearn.linear_model import Ridge
        alpha = 100 / (X_train.shape[1] * y_train.shape[1])
        reg = Ridge(solver="lsqr", alpha=alpha, random_state=random_state,
                    fit_intercept=False, max_iter=1000)
        reg.fit(X_train, y_train)
        preds_all = reg.predict(X_test)

    pearson_corrs, pearson_genes = [], []
    for i, gene in enumerate(genes):
        preds, targets = preds_all[:, i], y_test[:, i]
        if np.std(preds) < 1e-9 or np.std(targets) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(targets, preds)
            if np.isnan(corr):
                corr = 0.0
        pearson_corrs.append(corr)
        pearson_genes.append({"name": gene, "pearson_corr": corr})

    return {
        "pearson_corrs": pearson_genes,
        "pearson_mean": float(np.mean(pearson_corrs)),
        "pearson_std": float(np.std(pearson_corrs)),
        "n_train": len(y_train), "n_test": len(y_test),
    }


def evaluate_dataset(encoder, dataset_name, bench_data_root, embed_root, cfg):
    print(f"\n=== {dataset_name} ===")
    dataset_path = os.path.join(bench_data_root, dataset_name)
    embed_dir = os.path.join(embed_root, dataset_name, "spatia_image")
    os.makedirs(embed_dir, exist_ok=True)

    with open(os.path.join(dataset_path, "var_50genes.json")) as f:
        target_genes = json.load(f)["genes"]

    splits_dir = os.path.join(dataset_path, "splits")
    n_splits = len([f for f in os.listdir(splits_dir) if f.startswith("train_")])
    print(f"{len(target_genes)} target genes, {n_splits} CV splits")

    if encoder.model is None:
        encoder.load_model()

    split_results = []
    for split_idx in range(n_splits):
        train_df = pd.read_csv(os.path.join(splits_dir, f"train_{split_idx}.csv"))
        test_df = pd.read_csv(os.path.join(splits_dir, f"test_{split_idx}.csv"))

        split_assets = {}
        for split_key, split_df in [("train", train_df), ("test", test_df)]:
            embeddings_list, targets_list, available_genes = [], [], target_genes
            for i in range(len(split_df)):
                sample_id = split_df.iloc[i]["sample_id"]
                patches_path = os.path.join(dataset_path, split_df.iloc[i]["patches_path"])
                expr_path = os.path.join(dataset_path, split_df.iloc[i]["expr_path"])
                embed_path = os.path.join(embed_dir, f"{sample_id}_spatia_img.h5")

                if os.path.exists(embed_path):
                    with h5py.File(embed_path, "r") as f:
                        embeddings = f["embeddings"][:]
                        patch_barcodes = f["barcode"][:]
                else:
                    embeddings = encoder.extract_embeddings_from_h5(
                        patches_path, batch_size=cfg["batch_size"])
                    with h5py.File(patches_path, "r") as f:
                        patch_barcodes = f["barcode"][:]
                    with h5py.File(embed_path, "w") as f:
                        f.create_dataset("embeddings", data=embeddings)
                        f.create_dataset("barcode", data=patch_barcodes)

                adata = sc.read_h5ad(expr_path)
                if cfg["normalize"]:
                    adata.X = adata.X.astype(np.float64)
                    sc.pp.log1p(adata)
                available_genes = [g for g in target_genes if g in adata.var_names]
                target_expr = adata[:, available_genes].to_df()

                patch_barcodes = patch_barcodes.flatten()
                if len(patch_barcodes) > 0 and isinstance(patch_barcodes[0], bytes):
                    patch_barcodes = np.array([b.decode() for b in patch_barcodes])
                expr_barcodes = target_expr.index.values
                common = np.intersect1d(patch_barcodes, expr_barcodes)

                patch_idx = np.array([np.where(patch_barcodes == bc)[0][0] for bc in common])
                expr_idx = np.array([np.where(expr_barcodes == bc)[0][0] for bc in common])
                embeddings_list.append(embeddings[patch_idx])
                targets_list.append(target_expr.values[expr_idx])

            split_assets[split_key] = {
                "embeddings": np.vstack(embeddings_list),
                "targets": np.vstack(targets_list),
                "genes": available_genes,
            }

        X_train = split_assets["train"]["embeddings"]
        y_train = split_assets["train"]["targets"]
        X_test = split_assets["test"]["embeddings"]
        y_test = split_assets["test"]["targets"]
        genes = split_assets["train"]["genes"]

        if cfg["dimreduce"] == "PCA":
            latent_dim = min(cfg["latent_dim"], X_train.shape[1])
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=latent_dim, random_state=cfg["seed"])),
            ])
            X_train = pipe.fit_transform(X_train)
            X_test = pipe.transform(X_test)

        results = train_test_regression(
            X_train, X_test, y_train, y_test, genes=genes,
            method=cfg["method"], random_state=cfg["seed"])
        print(f"  split {split_idx}: PCC {results['pearson_mean']:.4f}")
        split_results.append(results)

    mean_per_split = [r["pearson_mean"] for r in split_results]
    gene_results = {}
    for result in split_results:
        for item in result["pearson_corrs"]:
            gene_results.setdefault(item["name"], []).append(item["pearson_corr"])
    gene_summary = sorted(
        [{"name": k, "mean": float(np.mean(v)), "std": float(np.std(v))}
         for k, v in gene_results.items()],
        key=lambda x: x["mean"], reverse=True)

    final = {
        "dataset": dataset_name,
        "pearson_mean": float(np.mean(mean_per_split)),
        "pearson_std": float(np.std(mean_per_split)),
        "mean_per_split": mean_per_split,
        "gene_results": gene_summary,
    }
    print(f"{dataset_name} final PCC: {final['pearson_mean']:.4f} +/- {final['pearson_std']:.4f}")
    return final


def main():
    args = parse_args()
    cfg = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "method": args.method,
        "dimreduce": args.dimreduce,
        "latent_dim": args.latent_dim,
        "normalize": not args.no_normalize,
    }
    os.makedirs(args.bench_data_root, exist_ok=True)
    os.makedirs(args.embed_root, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, datasets: {args.datasets}")

    if args.download:
        download_datasets(args.datasets, args.bench_data_root)

    encoder = SpatiaImageEncoder(args.checkpoint_dir)
    encoder.load_model()

    all_results = []
    for dataset_name in args.datasets:
        all_results.append(evaluate_dataset(
            encoder, dataset_name, args.bench_data_root, args.embed_root, cfg))

    overall_mean = float(np.mean([r["pearson_mean"] for r in all_results]))
    overall_std = float(np.std([r["pearson_mean"] for r in all_results]))
    print(f"\nOverall PCC: {overall_mean:.4f} +/- {overall_std:.4f}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"spatia_hest_results_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": "SPATIA",
            "encoder": "ViT-MAE (facebook/vit-mae-base)",
            "embedding_dim": encoder.embed_dim,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": cfg,
            "checkpoint": args.checkpoint_dir,
            "overall_pcc_mean": overall_mean,
            "overall_pcc_std": overall_std,
            "dataset_results": all_results,
        }, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
