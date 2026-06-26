# Prediction Tasks

Reproducible evaluation scripts for the clustering and cell-annotation benchmarks reported in the SPATIA paper (Section 6.1, Tables 2–4). All tasks run on frozen embeddings — **no GPU is needed once embeddings are extracted**.

| Task | Script | Data | Table |
|------|--------|------|-------|
| Cross-platform clustering (Xenium + CosMx) | `multi_seed_clustering.py` | `{Platform}_10K.h5ad` | 2 |
| Biomarker prediction (HEST) | `table3_hest_benchmark.py` | Auto-downloaded | 3 |
| scRNA-seq clustering (GSE155468) | `extract_spatia_embeddings.py` + `multi_seed_clustering.py` | `GSE155468.h5ad` | 4 |
| scRNA-seq clustering — all baselines | `table4_extract_and_eval.py` | `GSE155468.h5ad` | 4 |
| Cell annotation (GSE155468) | `annotation_eval.py` | `GSE155468.h5ad` | 4 |

---

## Table 2: Cross-Platform Clustering (Xenium + CosMx)

### Step 1 — Extract SPATIA embeddings

```bash
# Xenium (spatial mode — requires LMDB)
python prediction_tasks/scripts/extract_spatia_embeddings.py \
    --checkpoint /path/to/scprint_checkpoint.ckpt \
    --adata_path /path/to/Xenium_10K.h5ad \
    --lmdb_dir /path/to/xenium_images.lmdb \
    --output_path embeddings/spatia/Xenium_embeddings.npy

# CosMx
python prediction_tasks/scripts/extract_spatia_embeddings.py \
    --checkpoint /path/to/scprint_checkpoint.ckpt \
    --adata_path /path/to/CosMx_10K.h5ad \
    --lmdb_dir /path/to/cosmx_images.lmdb \
    --output_path embeddings/spatia/CosMx_embeddings.npy
```

Other model embeddings (PCA, scGPT, scFoundation, Nicheformer, UCE) follow the same `.npy` convention and should be placed at `{emb_dir}/{model}/{Platform}_embeddings.npy`.

### Step 2 — Run multi-seed clustering

```bash
cd prediction_tasks

python scripts/multi_seed_clustering.py \
    --task table2 \
    --table2_emb_dir /path/to/embeddings \
    --table2_data_dir /path/to/data \
    --output_dir results/multi_seed_eval \
    --n_seeds 5
```

**Evaluation protocol:** resolution sweep `[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]`, 5 seeds, no subsampling. Metrics: ARI and NMI (best over sweep per seed, mean ± std).

---

## Table 3: Biomarker Prediction (HEST Benchmark)

Uses SPATIA's ViT-MAE image encoder to predict gene expression from histology patches, following the HEST evaluation protocol (Jaume et al. 2024).

```bash
# GPU required. Data auto-downloads from HuggingFace on first run.
python prediction_tasks/scripts/table3_hest_benchmark.py \
    --download --datasets IDC \
    --batch-size 64 --method xgboost --dimreduce PCA --latent-dim 256
```

Pipeline: ViT-MAE patch embeddings (768-d) → PCA(256) → XGBoost → 50-HVG Pearson correlation.

Add more datasets: `--datasets IDC PAAD SKCM COAD LUAD`

**Expected results (IDC):**

| Model | PCC (mean ± std) |
|-------|-----------------|
| SPATIA (ViT-MAE) | 0.404 ± 0.012 |

---

## Table 4: scRNA-seq Clustering & Annotation (GSE155468)

**Dataset:** Li et al. 2020, GEO accession **GSE155468** — 48,082 cells, 11 cell types.

### Option A — SPATIA only

```bash
cd prediction_tasks

# Step 1: extract gene-only embeddings (no LMDB required)
python scripts/extract_spatia_embeddings.py \
    --checkpoint /path/to/scprint_checkpoint.ckpt \
    --adata_path /path/to/GSE155468.h5ad \
    --output_path embeddings/table4/spatia_embeddings.npy

# Step 2: cluster
python scripts/multi_seed_clustering.py \
    --task table4_clustering \
    --table4_emb_dir embeddings/table4 \
    --table4_data_dir /path/to/data \
    --output_dir results/table4_clustering \
    --n_seeds 5
```

### Option B — All baselines

```bash
# Extract all embeddings in one go
python scripts/table4_extract_and_eval.py \
    --step extract --model all \
    --data_path /path/to/GSE155468.h5ad \
    --output_dir results/table4_clustering \
    --scgpt_model_dir /path/to/scGPT/save/scGPT_human \
    --geneformer_model_dir /path/to/Geneformer-V2-316M \
    --cellplm_dir /path/to/CellPLM \
    --spatia_pkg_dir /path/to/gene_encoders/SPATIA-scgpt \
    --spatia_ckpt_dir /path/to/scgpt-checkpoint

# Run clustering
python scripts/table4_extract_and_eval.py \
    --step cluster \
    --data_path /path/to/GSE155468.h5ad \
    --output_dir results/table4_clustering \
    --n_seeds 5
```

**Evaluation protocol:** resolution sweep `[0.1, 0.2, ..., 1.4]`, 5 seeds, 2000-cell stratified subsample per seed.

**Expected results (Table 4):**

| Model | ARI | NMI |
|-------|-----|-----|
| SPATIA | 0.874 ± 0.022 | 0.846 ± 0.021 |
| scGPT | 0.845 ± 0.017 | 0.821 ± 0.011 |
| PCA | 0.832 ± 0.015 | 0.829 ± 0.018 |
| Geneformer | 0.479 ± 0.012 | 0.595 ± 0.023 |

### Cell Annotation

Supervised linear-probe annotation on frozen embeddings (train/test split, macro F1):

```bash
python prediction_tasks/scripts/annotation_eval.py \
    --emb_dir results/table4_clustering/embeddings \
    --output_dir results/annotation \
    --models pca scgpt geneformer spatia \
    --n_seeds 3 --test_size 0.2 --clf logreg
```

---

## Labels and Reproducibility

Both scripts resolve labels in this order:

1. `labels.csv` (column `celltype`) sitting next to the embeddings
2. The `celltype` column of `GSE155468.h5ad` (cached to `labels.csv` on first run)

The cell order in `labels.csv` must match the embedding row order.
