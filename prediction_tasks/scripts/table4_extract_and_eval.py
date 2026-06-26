
import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{os.environ.get('USER', 'user')}")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse, csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RESOLUTIONS = [r / 10 for r in range(1, 15)]
MODELS = ["pca", "cellplm", "scgpt", "geneformer", "spatia"]


def load_clustering_data(data_path: Path) -> ad.AnnData:
    print(f"Loading GSE155468 from {data_path} ...")
    if not data_path.exists():
        raise FileNotFoundError(
            f"GSE155468 not found at {data_path}. "
            "Download from GEO (accession GSE155468) and provide path via --data_path."
        )
    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique()

    if "celltype" not in adata.obs.columns:
        for alt in ("cell_type", "cellType", "annotation"):
            if alt in adata.obs.columns:
                adata.obs["celltype"] = adata.obs[alt]
                break
        else:
            raise ValueError(
                "Label column 'celltype' not found. "
                f"Available: {list(adata.obs.columns)}"
            )

    print(f"  {adata.shape}, {adata.obs['celltype'].nunique()} cell types")
    return adata


def resolve_clustering_labels(data_path: Path, emb_dir: Path):
    labels_csv = emb_dir / "labels.csv"
    if labels_csv.exists():
        labels = pd.read_csv(labels_csv)["celltype"].values
        print(f"Loaded {len(labels)} labels from {labels_csv}")
        return labels

    adata = load_clustering_data(data_path)
    labels = adata.obs["celltype"].values
    emb_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"celltype": labels}).to_csv(labels_csv, index=False)
    print(f"Cached labels to {labels_csv}")
    return labels


def extract_pca(adata: ad.AnnData, output_path: Path, **kwargs) -> np.ndarray:
    print("Extracting PCA embeddings ...")
    adata_copy = adata.copy()
    sc.pp.highly_variable_genes(
        adata_copy,
        n_top_genes=2000,
        flavor="seurat_v3",
        span=0.3 if adata_copy.n_obs < 10000 else 1.0,
    )
    adata_copy = adata_copy[:, adata_copy.var["highly_variable"]].copy()
    sc.pp.scale(adata_copy, max_value=10)
    sc.tl.pca(adata_copy, n_comps=50, svd_solver="arpack")
    embeddings = adata_copy.obsm["X_pca"]
    print(f"  Shape: {embeddings.shape} (from {adata_copy.n_vars} HVGs)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return embeddings


def extract_cellplm(
    adata: ad.AnnData,
    output_path: Path,
    device: str = "cuda",
    cellplm_dir: str | None = None,
    **kwargs,
) -> np.ndarray:
    if cellplm_dir is None:
        raise ValueError("--cellplm_dir is required for CellPLM extraction")
    print("Extracting CellPLM embeddings ...")
    cellplm_path = Path(cellplm_dir)
    sys.path.insert(0, str(cellplm_path))

    from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline

    pipeline = CellEmbeddingPipeline(
        pretrain_prefix="20230926_85M",
        pretrain_directory=str(cellplm_path / "ckpt"),
    )
    pred = pipeline.predict(
        adata.copy(),
        inference_config={"batch_size": 50000},
        device=device,
    )
    embeddings = pred.cpu().numpy()
    print(f"  Shape: {embeddings.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return embeddings


def extract_scgpt(
    adata: ad.AnnData,
    output_path: Path,
    device: str = "cuda",
    scgpt_model_dir: str | None = None,
    **kwargs,
) -> np.ndarray:
    if scgpt_model_dir is None:
        raise ValueError("--scgpt_model_dir is required for scGPT extraction")
    print("Extracting scGPT embeddings ...")
    model_dir = Path(scgpt_model_dir)
    scgpt_pkg_dir = model_dir.parent.parent
    sys.path.insert(0, str(scgpt_pkg_dir))

    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.tasks.cell_emb import get_batch_cell_embeddings

    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_dir / "args.json") as f:
        model_config = json.load(f)

    model_configs = {
        "embsize": model_config["embsize"],
        "nheads": model_config["nheads"],
        "d_hid": model_config["d_hid"],
        "nlayers": model_config["nlayers"],
        "pad_token": "<pad>",
        "pad_value": -2,
    }

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_config["embsize"],
        nhead=model_config["nheads"],
        d_hid=model_config["d_hid"],
        nlayers=model_config["nlayers"],
        nlayers_cls=3,
        n_cls=1,
        vocab=vocab,
        dropout=0.2,
        pad_token="<pad>",
        pad_value=-2,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        explicit_zero_prob=False,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        pre_norm=False,
    )

    state_dict = torch.load(model_dir / "best_model.pt", map_location=device)
    fixed = {k.replace("Wqkv.", "in_proj_"): v for k, v in state_dict.items()}
    model_dict = model.state_dict()
    filtered = {k: v for k, v in fixed.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    print(f"  Loaded {len(filtered)}/{len(state_dict)} params")

    adata_copy = adata.copy()
    gene_names = (
        adata_copy.var["gene_name"].tolist()
        if "gene_name" in adata_copy.var.columns
        else adata_copy.var_names.tolist()
    )
    adata_copy.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in gene_names]
    n_total = adata_copy.n_vars
    adata_copy = adata_copy[:, adata_copy.var["id_in_vocab"] >= 0].copy()
    print(f"  Matched {adata_copy.n_vars}/{n_total} genes")

    embeddings = get_batch_cell_embeddings(
        adata_copy,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=1200,
        batch_size=64,
        model_configs=model_configs,
        gene_ids=np.array(adata_copy.var["id_in_vocab"]),
    )
    print(f"  Shape: {embeddings.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return embeddings


def extract_geneformer(
    adata: ad.AnnData,
    output_path: Path,
    device: str = "cuda",
    geneformer_model_dir: str | None = None,
    geneformer_gene_mapping: str | None = None,
    **kwargs,
) -> np.ndarray:
    if geneformer_model_dir is None:
        raise ValueError("--geneformer_model_dir is required for Geneformer extraction")
    if geneformer_gene_mapping is None:
        raise ValueError("--geneformer_gene_mapping is required for Geneformer extraction")
    print("Extracting Geneformer embeddings ...")
    model_dir = Path(geneformer_model_dir)
    geneformer_pkg_dir = model_dir.parent
    sys.path.insert(0, str(geneformer_pkg_dir))

    from datasets import load_from_disk
    from geneformer import EmbExtractor, TranscriptomeTokenizer

    adata_copy = adata.copy()
    if "ensembl_id" not in adata_copy.var.columns:
        with open(geneformer_gene_mapping, "rb") as f:
            gene_mapping = pickle.load(f)
        ensembl_ids = []
        mapped = 0
        for gene in adata_copy.var_names:
            if gene in gene_mapping:
                ensembl_ids.append(gene_mapping[gene])
                mapped += 1
            else:
                ensembl_ids.append(gene)
        adata_copy.var["ensembl_id"] = ensembl_ids
        print(f"  Gene mapping: {mapped}/{len(adata_copy.var_names)}")

    X = adata_copy.X.toarray() if issparse(adata_copy.X) else adata_copy.X
    nonzero = X[X > 0]
    integer_pct = 100 * np.sum(nonzero == np.round(nonzero)) / len(nonzero) if len(nonzero) > 0 else 100
    if X.max() < 20 and integer_pct < 50:
        print("  Reversing log-transform ...")
        adata_copy.X = csr_matrix(np.expm1(X))

    X_raw = adata_copy.X.toarray() if issparse(adata_copy.X) else adata_copy.X
    adata_copy.obs["n_counts"] = X_raw.sum(axis=1)
    adata_copy.obs["_original_index"] = list(range(len(adata_copy)))

    temp_dir = tempfile.mkdtemp()
    try:
        temp_h5ad = os.path.join(temp_dir, "data.h5ad")
        safe_obs = adata_copy.obs[["_original_index", "n_counts"]].copy()
        if "celltype" in adata_copy.obs.columns:
            safe_obs["celltype"] = adata_copy.obs["celltype"]
        adata_clean = adata_copy.copy()
        adata_clean.obs = safe_obs
        adata_clean.write_h5ad(temp_h5ad)

        custom_attrs: dict[str, str] = {"_original_index": "_original_index"}
        if "celltype" in safe_obs.columns:
            custom_attrs["celltype"] = "celltype"

        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict=custom_attrs,
            nproc=4,
            model_input_size=4096,
            model_version="V2",
        )
        token_dir = os.path.join(temp_dir, "tokenized")
        os.makedirs(token_dir, exist_ok=True)
        tokenizer.tokenize_data(
            data_directory=temp_dir,
            output_directory=token_dir,
            output_prefix="data",
            file_format="h5ad",
        )

        tokenized_path = os.path.join(token_dir, "data.dataset")
        dataset = load_from_disk(tokenized_path)
        standard_fields = {"input_ids", "length", "lengths"}
        emb_labels = sorted(set(dataset.column_names) - standard_fields)

        embex = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cls",
            filter_data=None,
            max_ncells=None,
            emb_layer=-1,
            emb_label=emb_labels,
            forward_batch_size=24,
            model_version="V2",
            nproc=4,
        )
        emb_output_dir = os.path.join(temp_dir, "emb_output")
        os.makedirs(emb_output_dir, exist_ok=True)
        embs = embex.extract_embs(
            model_directory=str(model_dir),
            input_data_file=tokenized_path,
            output_directory=emb_output_dir,
            output_prefix="geneformer",
        )

        embedding_cols = [c for c in embs.columns if c not in emb_labels]
        embeddings = embs[embedding_cols].values
        if "_original_index" in embs.columns:
            order = embs["_original_index"].values.astype(int)
            embeddings = embeddings[np.argsort(order)]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"  Shape: {embeddings.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return embeddings


def extract_spatia(
    adata: ad.AnnData,
    output_path: Path,
    device: str = "cuda",
    spatia_pkg_dir: str | None = None,
    spatia_ckpt_dir: str | None = None,
    spatia_stats_dir: str | None = None,
    **kwargs,
) -> np.ndarray:
    if spatia_pkg_dir is None:
        raise ValueError("--spatia_pkg_dir is required for SPATIA extraction")
    if spatia_ckpt_dir is None:
        raise ValueError("--spatia_ckpt_dir is required for SPATIA extraction")
    print("Extracting SPATIA embeddings ...")
    pkg_dir = Path(spatia_pkg_dir)
    ckpt_dir = Path(spatia_ckpt_dir)
    stats_dir = Path(spatia_stats_dir) if spatia_stats_dir else None
    sys.path.insert(0, str(pkg_dir))

    from scgpt_spatial.model import TransformerModel
    from scgpt_spatial.tokenizer import GeneVocab
    from scgpt_spatial.tasks.cell_emb import get_batch_cell_embeddings

    vocab_path = ckpt_dir / "vocab.json"
    if not vocab_path.exists() and stats_dir is not None:
        vocab_path = stats_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"vocab.json not found in {ckpt_dir}"
            + (f" or {stats_dir}" if stats_dir else "")
        )
    vocab = GeneVocab.from_file(vocab_path)
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)

    state_dict_raw = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    actual_embsize = state_dict_raw["encoder.embedding.weight"].shape[1]
    actual_nheads = 8 if actual_embsize == 512 else 4
    layer_keys = [
        k for k in state_dict_raw
        if k.startswith("transformer_encoder.layers.") and ".self_attn." in k
    ]
    actual_nlayers = max(int(k.split(".")[2]) for k in layer_keys) + 1 if layer_keys else 4
    print(f"  Inferred: embsize={actual_embsize}, nlayers={actual_nlayers}, nheads={actual_nheads}")

    model_configs = {
        "embsize": actual_embsize,
        "nheads": actual_nheads,
        "d_hid": actual_embsize,
        "nlayers": actual_nlayers,
        "pad_token": "<pad>",
        "pad_value": -2,
    }

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=actual_embsize,
        nhead=actual_nheads,
        d_hid=actual_embsize,
        nlayers=actual_nlayers,
        nlayers_cls=3,
        n_cls=1,
        vocab=vocab,
        dropout=0.2,
        pad_token="<pad>",
        pad_value=-2,
        do_mvc=False,
        do_dab=False,
        use_generative_training=False,
        use_batch_labels=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        fast_transformer_backend=None,
        pre_norm=False,
    )

    remapped = {k.replace("Wqkv.", "in_proj_"): v for k, v in state_dict_raw.items()}
    model_dict = model.state_dict()
    filtered = {k: v for k, v in remapped.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    print(f"  Loaded {len(filtered)}/{len(state_dict_raw)} params")

    adata_copy = adata.copy()
    X = adata_copy.X.toarray() if issparse(adata_copy.X) else np.asarray(adata_copy.X)
    nz = X[X > 0]
    is_integer = nz.size == 0 or np.allclose(nz, np.round(nz))
    if X.max() < 20 and not is_integer:
        print("  Detected log1p data -> reversing with expm1")
        X = np.expm1(X)
    else:
        print("  Detected raw counts -> no expm1")
    adata_copy.X = csr_matrix(X)

    gene_names = (
        adata_copy.var["gene_name"].tolist()
        if "gene_name" in adata_copy.var.columns
        else adata_copy.var_names.tolist()
    )
    adata_copy.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in gene_names]
    n_total = adata_copy.n_vars
    adata_copy = adata_copy[:, adata_copy.var["id_in_vocab"] >= 0].copy()
    print(f"  Matched {adata_copy.n_vars}/{n_total} genes")

    gene_stats_file: str | None = None
    if stats_dir is not None:
        candidate = stats_dir / "all_dict_mean_std.csv"
        if candidate.exists():
            gene_stats_file = str(candidate)

    embeddings = get_batch_cell_embeddings(
        adata_copy,
        gene_stats_dict_file=gene_stats_file,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=1200,
        batch_size=64,
        model_configs=model_configs,
        gene_ids=np.array(adata_copy.var["id_in_vocab"]),
    )
    print(f"  Shape: {embeddings.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return embeddings


EXTRACTORS = {
    "pca": extract_pca,
    "cellplm": extract_cellplm,
    "scgpt": extract_scgpt,
    "geneformer": extract_geneformer,
    "spatia": extract_spatia,
}


def run_multi_seed_clustering(
    emb_dir: Path,
    data_path: Path,
    output_dir: Path,
    n_seeds: int = 5,
    models: list[str] = MODELS,
) -> pd.DataFrame:
    print("=" * 70)
    print("Table 4: Multi-seed Clustering (GSE155468 dataset)")
    print("=" * 70)

    labels = resolve_clustering_labels(data_path, emb_dir)
    valid_mask = pd.notna(labels)
    if not valid_mask.all():
        labels = labels[valid_mask]
        print(f"  Filtered {(~valid_mask).sum()} NaN labels -> {len(labels)} cells remain")

    rows = []
    all_results: dict = {}
    n_sample = 2000

    for model in models:
        emb_path = emb_dir / f"{model}_embeddings.npy"
        if not emb_path.exists():
            print(f"  SKIP {model}: {emb_path} not found")
            continue

        embeddings = np.load(emb_path)
        if not valid_mask.all():
            embeddings = embeddings[valid_mask]

        print(f"\n  {model} (d={embeddings.shape[1]}, n={len(labels)}) ...")
        seed_results = []

        for seed in range(n_seeds):
            if len(labels) > n_sample:
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sample, random_state=seed)
                idx, _ = next(sss.split(embeddings, labels))
            else:
                idx = np.arange(len(labels))

            emb_s = embeddings[idx]
            lab_s = labels[idx]

            adata_tmp = ad.AnnData(X=emb_s)
            adata_tmp.obsm["X_emb"] = emb_s
            adata_tmp.obs["gt"] = lab_s
            sc.pp.neighbors(adata_tmp, use_rep="X_emb", n_neighbors=15, random_state=seed)

            best_ari, best_nmi = -1.0, -1.0
            for res in RESOLUTIONS:
                sc.tl.leiden(adata_tmp, resolution=res, key_added="leiden", random_state=seed)
                cl = adata_tmp.obs["leiden"].values.astype(str)
                gt = adata_tmp.obs["gt"].values.astype(str)
                ari = adjusted_rand_score(gt, cl)
                nmi = normalized_mutual_info_score(gt, cl)
                if ari > best_ari:
                    best_ari = ari
                if nmi > best_nmi:
                    best_nmi = nmi

            seed_results.append({"seed": seed, "ari": best_ari, "nmi": best_nmi})

        aris = [r["ari"] for r in seed_results]
        nmis = [r["nmi"] for r in seed_results]
        result = {
            "per_seed": seed_results,
            "ari_mean": float(np.mean(aris)),
            "ari_std": float(np.std(aris)),
            "nmi_mean": float(np.mean(nmis)),
            "nmi_std": float(np.std(nmis)),
        }
        all_results[model] = result
        rows.append({
            "Model": model,
            "ARI": f"{result['ari_mean']:.4f} +/- {result['ari_std']:.4f}",
            "NMI": f"{result['nmi_mean']:.4f} +/- {result['nmi_std']:.4f}",
            "ARI_mean": result["ari_mean"],
            "ARI_std": result["ari_std"],
            "NMI_mean": result["nmi_mean"],
            "NMI_std": result["nmi_std"],
        })
        print(f"    ARI = {result['ari_mean']:.4f} +/- {result['ari_std']:.4f}")
        print(f"    NMI = {result['nmi_mean']:.4f} +/- {result['nmi_std']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "table4_clustering_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table4_clustering_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("Table 4 Clustering Summary")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"  {row['Model']:15s}  ARI={row['ARI']}  NMI={row['NMI']}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table 4: embedding extraction and multi-seed clustering (GSE155468)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--step", required=True, choices=["extract", "cluster", "all"])
    parser.add_argument("--model", default="all",
                        help=f"Which model(s) to run: {MODELS} or 'all'")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_seeds", type=int, default=5)

    parser.add_argument("--data_path", default=None,
                        help="Path to GSE155468.h5ad")
    parser.add_argument("--output_dir", required=True,
                        help="Root directory for embeddings/ subdir and result files")

    parser.add_argument("--cellplm_dir", default=None,
                        help="CellPLM installation dir (must contain CellPLM/ package and ckpt/)")

    parser.add_argument("--scgpt_model_dir", default=None,
                        help="scGPT model directory (contains best_model.pt, vocab.json, args.json)")

    parser.add_argument("--geneformer_model_dir", default=None,
                        help="Geneformer-V2-316M model directory")
    parser.add_argument("--geneformer_gene_mapping", default=None,
                        help="Path to gene_name_id_dict.pkl for gene symbol -> Ensembl ID mapping")

    parser.add_argument("--spatia_pkg_dir", default=None,
                        help="SPATIA-scgpt source directory (contains scgpt_spatial/)")
    parser.add_argument("--spatia_ckpt_dir", default=None,
                        help="SPATIA checkpoint directory (contains best_model.pt)")
    parser.add_argument("--spatia_stats_dir", default=None,
                        help="Directory with all_dict_mean_std.csv and fallback vocab.json")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    emb_dir = output_dir / "embeddings"
    data_path = Path(args.data_path) if args.data_path else None

    if args.step in ("extract", "all") and data_path is None:
        parser.error("--data_path is required for the extract step")
    if args.step == "cluster" and data_path is None and not (emb_dir / "labels.csv").exists():
        parser.error("--data_path is required (no labels.csv found next to embeddings)")

    extractor_kwargs = {
        "device": args.device,
        "cellplm_dir": args.cellplm_dir,
        "scgpt_model_dir": args.scgpt_model_dir,
        "geneformer_model_dir": args.geneformer_model_dir,
        "geneformer_gene_mapping": args.geneformer_gene_mapping,
        "spatia_pkg_dir": args.spatia_pkg_dir,
        "spatia_ckpt_dir": args.spatia_ckpt_dir,
        "spatia_stats_dir": args.spatia_stats_dir,
    }

    models = MODELS if args.model == "all" else [args.model]

    if args.step in ("extract", "all"):
        adata = load_clustering_data(data_path)
        labels_csv = emb_dir / "labels.csv"
        if not labels_csv.exists():
            emb_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"celltype": adata.obs["celltype"].values}).to_csv(labels_csv, index=False)
            print(f"  Cached labels to {labels_csv}")
        for model in models:
            emb_path = emb_dir / f"{model}_embeddings.npy"
            if emb_path.exists():
                print(f"  {model}: embeddings already exist, skipping. Delete {emb_path} to re-extract.")
                continue
            try:
                EXTRACTORS[model](adata, emb_path, **extractor_kwargs)
            except Exception as e:
                import traceback
                print(f"  ERROR extracting {model}: {e}")
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    if args.step in ("cluster", "all"):
        run_multi_seed_clustering(
            emb_dir=emb_dir,
            data_path=data_path,
            output_dir=output_dir,
            n_seeds=args.n_seeds,
            models=models,
        )


if __name__ == "__main__":
    main()
