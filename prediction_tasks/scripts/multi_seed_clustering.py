
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{os.environ.get('USER', 'user')}")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

TABLE2_MODELS = ["pca", "scgpt", "scfoundation", "nicheformer", "uce", "spatia"]
TABLE2_PLATFORMS = ["Xenium", "CosMx"]
TABLE2_LABEL_COL = "annotation"

TABLE4_MODELS = ["pca", "cellplm", "scgpt", "geneformer", "spatia"]
TABLE4_LABEL_COL = "celltype"

TABLE2_RESOLUTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
TABLE4_RESOLUTIONS = [r / 10 for r in range(1, 15)]


def run_leiden_single_seed(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int,
    resolutions: list = TABLE4_RESOLUTIONS,
    n_neighbors: int = 15,
    pca_dim: int = 50,
) -> dict:
    from sklearn.decomposition import PCA

    adata = ad.AnnData(X=embeddings)
    if embeddings.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        adata.obsm["X_emb"] = pca.fit_transform(embeddings)
    else:
        adata.obsm["X_emb"] = embeddings
    adata.obs["ground_truth"] = labels

    sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=n_neighbors, random_state=seed)

    best_ari = -1.0
    best_nmi = -1.0
    best_res_ari = 0.0
    best_res_nmi = 0.0

    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, key_added="leiden", random_state=seed)
        clusters = adata.obs["leiden"].values.astype(str)
        gt = adata.obs["ground_truth"].values.astype(str)

        ari = adjusted_rand_score(gt, clusters)
        nmi = normalized_mutual_info_score(gt, clusters)

        if ari > best_ari:
            best_ari = ari
            best_res_ari = res
        if nmi > best_nmi:
            best_nmi = nmi
            best_res_nmi = res

    return {
        "seed": seed,
        "best_ari": best_ari,
        "best_nmi": best_nmi,
        "best_res_ari": best_res_ari,
        "best_res_nmi": best_res_nmi,
    }


def run_multi_seed(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_seeds: int = 5,
    seed_start: int = 0,
    resolutions: list = TABLE4_RESOLUTIONS,
) -> dict:
    seeds = list(range(seed_start, seed_start + n_seeds))
    all_results = []

    for i, s in enumerate(seeds):
        print(f"      seed {s} ({i+1}/{n_seeds}) ...", end=" ", flush=True)
        result = run_leiden_single_seed(embeddings, labels, seed=s, resolutions=resolutions)
        print(f"ARI={result['best_ari']:.4f} NMI={result['best_nmi']:.4f}")
        all_results.append(result)

    aris = [r["best_ari"] for r in all_results]
    nmis = [r["best_nmi"] for r in all_results]

    return {
        "per_seed": all_results,
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "nmi_mean": float(np.mean(nmis)),
        "nmi_std": float(np.std(nmis)),
        "n_seeds": n_seeds,
    }


def run_table2(
    n_seeds: int,
    emb_dir: Path,
    data_dir: Path,
    output_dir: Path,
    models: list[str] = TABLE2_MODELS,
) -> pd.DataFrame:
    print("=" * 70)
    print("Table 2: Cross-platform Clustering (Xenium / CosMx)")
    print(f"  Seeds: {n_seeds}  |  emb_dir: {emb_dir}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}
    rows = []

    for platform in TABLE2_PLATFORMS:
        data_path = data_dir / f"{platform}_10K.h5ad"
        if not data_path.exists():
            print(f"  SKIP {platform}: data not found at {data_path}")
            continue

        adata = sc.read_h5ad(data_path)
        labels = adata.obs[TABLE2_LABEL_COL].values
        valid_mask = pd.notna(labels)
        if not valid_mask.all():
            labels = labels[valid_mask]
            print(f"  {platform}: filtered {(~valid_mask).sum()} NaN labels")

        for model in models:
            emb_path = emb_dir / model / f"{platform}_embeddings.npy"
            if not emb_path.exists():
                print(f"  SKIP {model}/{platform}: {emb_path} not found")
                continue

            embeddings = np.load(emb_path)
            if not valid_mask.all():
                embeddings = embeddings[valid_mask]

            print(f"\n  {model}/{platform} (d={embeddings.shape[1]}, n={len(labels)}) ...")
            result = run_multi_seed(
                embeddings, labels, n_seeds=n_seeds, resolutions=TABLE2_RESOLUTIONS
            )
            all_results[f"{platform}/{model}"] = result
            rows.append({
                "Platform": platform,
                "Model": model,
                "ARI": f"{result['ari_mean']:.4f} ± {result['ari_std']:.4f}",
                "NMI": f"{result['nmi_mean']:.4f} ± {result['nmi_std']:.4f}",
                "ARI_mean": result["ari_mean"],
                "ARI_std": result["ari_std"],
                "NMI_mean": result["nmi_mean"],
                "NMI_std": result["nmi_std"],
            })
            print(f"    ARI = {result['ari_mean']:.4f} ± {result['ari_std']:.4f}")
            print(f"    NMI = {result['nmi_mean']:.4f} ± {result['nmi_std']:.4f}")

    with open(output_dir / "table2_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table2_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("Table 2 Summary")
    print("=" * 70)
    for platform in TABLE2_PLATFORMS:
        sub = df[df["Platform"] == platform]
        if sub.empty:
            continue
        print(f"\n  {platform}:")
        for _, row in sub.iterrows():
            print(f"    {row['Model']:15s}  ARI={row['ARI']}  NMI={row['NMI']}")

    return df


def load_table4_labels(data_dir: Path, emb_dir: Path) -> np.ndarray:
    labels_csv = emb_dir / "labels.csv"
    if labels_csv.exists():
        labels = pd.read_csv(labels_csv)["celltype"].values
        print(f"  Loaded {len(labels)} labels from {labels_csv}")
        return labels

    data_path = data_dir / "GSE155468.h5ad"
    if not data_path.exists():
        raise FileNotFoundError(
            f"No labels.csv next to embeddings and GSE155468.h5ad not found at {data_path}. "
            "Download from GEO (accession GSE155468) and place it there."
        )
    adata = sc.read_h5ad(data_path)
    label_col = TABLE4_LABEL_COL
    if label_col not in adata.obs.columns:
        for alt in ("cell_type", "cellType", "annotation"):
            if alt in adata.obs.columns:
                label_col = alt
                break
        else:
            raise ValueError(
                f"Label column '{TABLE4_LABEL_COL}' not found. "
                f"Available: {list(adata.obs.columns)}"
            )
    labels = adata.obs[label_col].values
    emb_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"celltype": labels}).to_csv(labels_csv, index=False)
    print(f"  Cached labels to {labels_csv}")
    return labels


def run_table4_clustering(
    n_seeds: int,
    emb_dir: Path,
    data_dir: Path,
    output_dir: Path,
    models: list[str] = TABLE4_MODELS,
) -> pd.DataFrame:
    print("=" * 70)
    print("Table 4: Clustering (GSE155468, CellPLM evaluation protocol)")
    print(f"  Seeds: {n_seeds}  |  resolutions: {TABLE4_RESOLUTIONS}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}
    rows = []

    labels = load_table4_labels(data_dir, emb_dir)
    valid_mask = pd.notna(labels)
    if not valid_mask.all():
        labels = labels[valid_mask]
        print(f"  Filtered {(~valid_mask).sum()} NaN labels → {len(labels)} cells remain")

    for model in models:
        emb_path = emb_dir / f"{model}_embeddings.npy"
        if not emb_path.exists():
            print(f"  SKIP {model}: {emb_path} not found")
            continue

        embeddings = np.load(emb_path)
        if not valid_mask.all():
            embeddings = embeddings[valid_mask]

        print(f"\n  {model} (d={embeddings.shape[1]}, n={len(labels)}) ...")
        result = run_multi_seed(
            embeddings, labels, n_seeds=n_seeds, resolutions=TABLE4_RESOLUTIONS
        )
        all_results[model] = result
        rows.append({
            "Model": model,
            "ARI": f"{result['ari_mean']:.4f} ± {result['ari_std']:.4f}",
            "NMI": f"{result['nmi_mean']:.4f} ± {result['nmi_std']:.4f}",
            "ARI_mean": result["ari_mean"],
            "ARI_std": result["ari_std"],
            "NMI_mean": result["nmi_mean"],
            "NMI_std": result["nmi_std"],
        })
        print(f"    ARI = {result['ari_mean']:.4f} ± {result['ari_std']:.4f}")
        print(f"    NMI = {result['nmi_mean']:.4f} ± {result['nmi_std']:.4f}")

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
        description="Multi-seed Leiden clustering evaluation (Tables 2 & 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True,
        choices=["table2", "table4_clustering", "all"],
        help="Which evaluation to run",
    )
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds")

    parser.add_argument(
        "--table2_emb_dir", type=str, default=None,
        help="Root dir with Table 2 embeddings: {dir}/{model}/{Platform}_embeddings.npy",
    )
    parser.add_argument(
        "--table2_data_dir", type=str, default=None,
        help="Dir with {Platform}_10K.h5ad files for Table 2",
    )
    parser.add_argument(
        "--table2_models", nargs="+", default=None,
        help="Subset of models to evaluate for Table 2 (default: all)",
    )

    parser.add_argument(
        "--table4_emb_dir", type=str, default=None,
        help="Dir with {model}_embeddings.npy files for Table 4",
    )
    parser.add_argument(
        "--table4_data_dir", type=str, default=None,
        help="Dir containing GSE155468.h5ad for Table 4",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write result JSON and CSV files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.task in ("table2", "all"):
        if not args.table2_emb_dir or not args.table2_data_dir:
            parser.error("--table2_emb_dir and --table2_data_dir are required for table2")
        run_table2(
            args.n_seeds,
            emb_dir=Path(args.table2_emb_dir),
            data_dir=Path(args.table2_data_dir),
            output_dir=output_dir,
            models=args.table2_models or TABLE2_MODELS,
        )

    if args.task in ("table4_clustering", "all"):
        if not args.table4_emb_dir:
            parser.error("--table4_emb_dir is required for table4_clustering")
        emb_dir = Path(args.table4_emb_dir)
        has_labels = (emb_dir / "labels.csv").exists()
        if not args.table4_data_dir and not has_labels:
            parser.error(
                "--table4_data_dir is required for table4_clustering "
                "(no labels.csv found next to embeddings)"
            )
        run_table4_clustering(
            args.n_seeds,
            emb_dir=emb_dir,
            data_dir=Path(args.table4_data_dir) if args.table4_data_dir else emb_dir,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
