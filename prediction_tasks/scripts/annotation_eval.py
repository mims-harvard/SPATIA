
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score

os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{os.environ.get('USER', 'user')}")

MODELS = ["pca", "cellplm", "scgpt", "geneformer", "spatia"]


def load_labels(emb_dir, data_path, labels_path):
    if labels_path is None:
        for cand in (emb_dir / "labels.csv", emb_dir / "labels.npy"):
            if cand.exists():
                labels_path = cand
                break

    if labels_path is not None:
        labels_path = Path(labels_path)
        if labels_path.suffix == ".npy":
            return np.load(labels_path, allow_pickle=True)
        df = pd.read_csv(labels_path)
        col = "celltype" if "celltype" in df.columns else df.columns[-1]
        return df[col].values

    if data_path is None:
        raise ValueError("Provide --labels_path or --data_path for cell-type labels.")

    import scanpy as sc
    adata = sc.read_h5ad(data_path)
    for col in ("celltype", "cell_type", "cellType", "annotation"):
        if col in adata.obs.columns:
            return adata.obs[col].values
    raise ValueError(f"No celltype column in {data_path}. Have: {list(adata.obs.columns)}")


def build_classifier(kind, seed):
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=seed)
    return LogisticRegression(max_iter=2000, n_jobs=-1, random_state=seed)


def evaluate_model(embeddings, labels, n_seeds, test_size, clf_kind):
    per_seed = []
    for seed in range(n_seeds):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(embeddings, labels))
        clf = build_classifier(clf_kind, seed)
        clf.fit(embeddings[train_idx], labels[train_idx])
        pred = clf.predict(embeddings[test_idx])
        f1 = f1_score(labels[test_idx], pred, average="macro", zero_division=0)
        prec = precision_score(labels[test_idx], pred, average="macro", zero_division=0)
        per_seed.append({"seed": seed, "f1": float(f1), "precision": float(prec)})

    f1s = [r["f1"] for r in per_seed]
    precs = [r["precision"] for r in per_seed]
    return {
        "per_seed": per_seed,
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(precs)),
        "precision_std": float(np.std(precs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Cell annotation evaluation on frozen embeddings")
    parser.add_argument("--emb_dir", required=True, help="Dir with {model}_embeddings.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_path", default=None, help="GSE155468.h5ad (for labels)")
    parser.add_argument("--labels_path", default=None, help="labels.csv or labels.npy")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--clf", choices=["logreg", "mlp"], default="logreg")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(emb_dir, args.data_path, args.labels_path)
    labels = np.asarray(labels)
    valid = pd.notna(labels)
    if not valid.all():
        print(f"  dropping {(~valid).sum()} cells with missing labels")

    results = {}
    rows = []
    for model in args.models:
        emb_path = emb_dir / f"{model}_embeddings.npy"
        if not emb_path.exists():
            print(f"  SKIP {model}: {emb_path} not found")
            continue

        emb = np.load(emb_path)
        y = labels
        if not valid.all():
            emb, y = emb[valid], labels[valid]

        print(f"\n  {model} (d={emb.shape[1]}, n={len(y)}) ...")
        res = evaluate_model(emb, y, args.n_seeds, args.test_size, args.clf)
        results[model] = res
        rows.append({
            "Model": model,
            "F1": f"{res['f1_mean']:.4f} +/- {res['f1_std']:.4f}",
            "Precision": f"{res['precision_mean']:.4f} +/- {res['precision_std']:.4f}",
            "F1_mean": res["f1_mean"],
            "Precision_mean": res["precision_mean"],
        })
        print(f"    F1        = {res['f1_mean']:.4f} +/- {res['f1_std']:.4f}")
        print(f"    Precision = {res['precision_mean']:.4f} +/- {res['precision_std']:.4f}")

    with open(output_dir / "annotation_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(rows).to_csv(output_dir / "annotation_summary.csv", index=False)
    print(f"\nWrote {output_dir}/annotation_summary.csv")


if __name__ == "__main__":
    main()
