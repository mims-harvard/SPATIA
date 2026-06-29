# SPATIA Prediction Tasks

Reproducible evaluation scripts for the clustering and cell-annotation
benchmarks reported in the SPATIA paper (Section 6.1, Tables 2 & 4).

## Installation

**Full install** (GPU embedding extraction + evaluation):

```bash
# From the SPATIA root directory
pip install -e data_processing/                # data loader (install first)
pip install -e gene_encoders/SPATIA-scprint/   # representation model
pip install lmdb tifffile scikit-image transformers  # image processing deps
```

**Lightweight install** (evaluation only, no GPU needed):

```bash
pip install scanpy leidenalg scikit-learn anndata pandas numpy
```

**Environment setup** (required on shared filesystems):

```bash
# Prevents numba cache write errors on read-only site-packages
export NUMBA_CACHE_DIR=/tmp/numba_cache_$USER && mkdir -p $NUMBA_CACHE_DIR
```

**Python version**: Python 3.10 is required (`scdataloader` requires `>=3.10,<3.11`).

**Image preprocessor note**: The data loader uses `AutoImageProcessor` from
HuggingFace `transformers`, which is compatible with both CLIP and ViT-MAE models.
If using spatial mode (with LMDB images), the preprocessor converts 256x256
grayscale crops into `(3, 224, 224)` float tensors for the ViT-MAE encoder.

## Labels and reproducibility

Clustering and annotation need per-cell type labels. Both scripts resolve
labels in this order:

1. a `labels.csv` (column `celltype`) sitting next to the embeddings, then
2. the `celltype` column of `GSE155468.h5ad` (and they cache it to `labels.csv`).

The shipped `results/table4_clustering/embeddings/` ships the embedding `.npy`
files. If a matching `labels.csv` is present, Table 4 clustering/annotation
reproduce **fully offline** from the shipped embeddings. If you only have the
h5ad, pass `--table4_data_dir` (clustering) or `--data_path` (annotation) and
the first run will cache `labels.csv` for you. The cell order in `labels.csv`
must match the order of the embedding rows.

---

## Cross-Platform Clustering (Xenium + CosMx)

### Step 1 — Extract SPATIA embeddings

```bash
# Xenium
python scripts/extract_spatia_embeddings.py \
    --checkpoint /path/to/scprint_checkpoint.ckpt \
    --adata_path /path/to/Xenium_10K.h5ad \
    --lmdb_dir /path/to/xenium_images.lmdb \
    --output_path embeddings/spatia/Xenium_embeddings.npy

# CosMx
python scripts/extract_spatia_embeddings.py \
    --checkpoint /path/to/scprint_checkpoint.ckpt \
    --adata_path /path/to/CosMx_10K.h5ad \
    --lmdb_dir /path/to/cosmx_images.lmdb \
    --output_path embeddings/spatia/CosMx_embeddings.npy
```

Other model embeddings (PCA, scGPT, scFoundation, Nicheformer, UCE) follow the same
`.npy` convention and should be placed under `{emb_dir}/{model}/{Platform}_embeddings.npy`.

### Step 2 — Run multi-seed clustering

```bash
python scripts/multi_seed_clustering.py \
    --task table2 \
    --table2_emb_dir /path/to/embeddings \
    --table2_data_dir /path/to/data \
    --output_dir results/multi_seed_eval \
    --n_seeds 5
```

**Data layout:**
```
{table2_data_dir}/
    Xenium_10K.h5ad    # obs["annotation"] = cell type labels
    CosMx_10K.h5ad
{table2_emb_dir}/
    {model}/
        Xenium_embeddings.npy
        CosMx_embeddings.npy
```

**Expected outputs:** `results/multi_seed_eval/table2_summary.csv`

---

## Biomarker Prediction (HEST Benchmark)

Uses SPATIA's ViT-MAE image encoder to predict gene expression from
histology patches, following the HEST evaluation protocol (Jaume et al. 2024).

```bash
# GPU required. Data auto-downloads from HuggingFace on first run.
python scripts/hest_benchmark.py \
    --download --datasets IDC \
    --batch-size 64 --method xgboost --dimreduce PCA --latent-dim 256
```

Pipeline: ViT-MAE patch embeddings (768-d) -> PCA(256) -> XGBoost -> 50-HVG Pearson correlation.

Add more datasets as needed: `--datasets IDC PAAD SKCM COAD LUAD`.

**Expected output:** `{results_dir}/spatia_hest_results_*.json` with per-gene and overall PCC.

---

## scRNA-seq Clustering (GSE155468)

Dataset: Li et al. 2020, GEO accession **GSE155468** — 48,082 cells, 11 cell types.
Download the processed `.h5ad` from GEO and place it at `GSE155468.h5ad`.

### Option A — SPATIA only (extract + cluster)

```bash
# Step 1: Extract SPATIA-scprint embeddings (gene-only mode, no LMDB)
python scripts/extract_spatia_embeddings.py \
    --checkpoint /path/to/scprint_checkpoint.ckpt \
    --adata_path /path/to/GSE155468.h5ad \
    --output_path embeddings/table4/spatia_embeddings.npy

# Step 2: Run clustering
python scripts/multi_seed_clustering.py \
    --task table4_clustering \
    --table4_emb_dir embeddings/table4 \
    --table4_data_dir /path/to/data \
    --output_dir results/table4_clustering \
    --n_seeds 5
```

### Option B — All baselines (PCA, CellPLM, scGPT, Geneformer, SPATIA)

```bash
# Extract all embeddings in one go
python scripts/extract_and_eval.py \
    --step extract --model all \
    --data_path /path/to/GSE155468.h5ad \
    --output_dir results/table4_clustering \
    --scgpt_model_dir /path/to/scGPT/save/scGPT_human \
    --geneformer_model_dir /path/to/Geneformer-V2-316M \
    --geneformer_gene_mapping /path/to/geneformer_001/geneformer/gene_name_id_dict.pkl \
    --cellplm_dir /path/to/CellPLM \
    --spatia_pkg_dir /path/to/gene_encoders/SPATIA-scgpt \
    --spatia_ckpt_dir /path/to/scgpt-train-YYYYMMDD-... \
    --spatia_stats_dir /path/to/scGPT-spatial/checkpoints/scGPT_spatial_v1

# Run multi-seed clustering (2000-cell stratified subsample, resolution sweep 0.1-1.4)
python scripts/extract_and_eval.py \
    --step cluster \
    --data_path /path/to/GSE155468.h5ad \
    --output_dir results/table4_clustering \
    --n_seeds 5
```

---

## Cell Annotation 

Supervised linear-probe annotation on frozen embeddings: train a classifier on
a stratified train split and report macro F1 and macro precision on the held-out
test split, averaged over seeds.

```bash
python scripts/annotation_eval.py \
    --emb_dir results/table4_clustering/embeddings \
    --output_dir results/annotation \
    --models pca scgpt geneformer spatia \
    --n_seeds 3 --test_size 0.2 --clf logreg
```

If `labels.csv` is not next to the embeddings, add `--data_path /path/to/GSE155468.h5ad`.

**Expected outputs:** `results/annotation/annotation_summary.csv` (F1, precision per model).

---

## Evaluation Protocol

### Spatial clustering
- Resolution sweep: `[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]`
- Seeds: 5, no subsampling
- Metrics: ARI and NMI (best over resolution sweep per seed, then mean ± std)

### HEST biomarker prediction
- Encoder: ViT-MAE base (768-d patch embeddings)
- Pipeline: PCA(256) -> XGBoost -> 50-HVG Pearson correlation
- Datasets: IDC (and optionally PAAD, SKCM, COAD, LUAD)
- Data auto-downloaded from HuggingFace (`MahmoodLab/hest-bench`)

### scRNA-seq clustering, CellPLM protocol
- Resolution sweep: `[0.1, 0.2, ..., 1.4]`
- Seeds: 5, 2000-cell stratified subsample per seed
- Metrics: ARI and NMI (best over resolution sweep per seed, then mean ± std)

---