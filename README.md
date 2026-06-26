<h1 align="center">
  SPATIA: Multimodal Model for Prediction and Generation of Spatial Cell Phenotypes
</h1>

<p align="center">
  <a href="https://zitniklab.hms.harvard.edu/SPATIA/"><img src="https://img.shields.io/badge/Website-SPATIA-4CAF50?logo=googlechrome&logoColor=white" /></a>
  <a href="https://arxiv.org/abs/2507.04704"><img src="https://img.shields.io/badge/Paper-arXiv%202507.04704-b31b1b?logo=arxiv&logoColor=white" /></a>
  <a href="https://huggingface.co/mims-harvard"><img src="https://img.shields.io/badge/🤗%20HuggingFace-mims--harvard-FFD21E" /></a>
  <a href="https://mims-harvard.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Docs-readthedocs-blue?logo=readthedocs&logoColor=white" /></a>
</p>

## Overview

Understanding how cellular morphology, gene expression, and spatial organization jointly shape tissue function is a central challenge in biology. Image-based spatial transcriptomics now provides high-resolution measurements of cell images and gene expression, but most methods analyze these modalities in isolation.

**SPATIA** is a multi-scale model for spatial transcriptomics that:
- Learns cell-level embeddings by fusing image-derived morphological tokens and transcriptomic tokens via cross-attention
- Aggregates embeddings at niche and tissue levels with transformer modules to capture spatial context
- Generates cell morphology images conditioned on predicted state transitions using flow matching

<p align="center"> <img src="static/images/overview_view.png" width="500" align="center"> </p>

---

## Installation

**Requirements:** Python 3.10, CUDA 12.1+, GPU with 40GB+ VRAM (A100/H100 recommended).

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate spatia

# Or manually:
# conda create -n spatia python=3.10
# conda activate spatia
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
#     --index-url https://download.pytorch.org/whl/cu121

# 2. Install Flash Attention (required for the scPRINT transformer)
pip install flash-attn --no-build-isolation

# 3. Install data processing package (must be installed BEFORE scprint)
pip install -e data_processing

# 4. Install SPATIA-scprint
pip install -e gene_encoders/SPATIA-scprint
```

<!-- ### Environment notes

- **Python version**: The `scdataloader` package
  requires `>=3.10,<3.11` and `lamindb==0.76.12` requires 3.10.
- **lamindb**: Required for pretraining data management. After install, initialize
  with `lamin init --storage ./data --name scprint_db` (only needed once).
- **bionty**: Installed automatically with `lamindb[bionty]` via `data_processing`.
  After first install, run `bionty.base.reset_sources()` to initialize the
  ontology tables.
- **Flash Attention**: Requires a compatible CUDA toolkit. If `flash-attn` fails
  to build, the scGPT path has a built-in fallback to standard PyTorch
  `nn.MultiheadAttention` (slower but functional).
- **numba cache**: On shared filesystems where site-packages is read-only,
  export a writable cache before running anything that imports scanpy:
  ```bash
  export NUMBA_CACHE_DIR=/tmp/numba_cache_$USER && mkdir -p $NUMBA_CACHE_DIR
  ```
- **Image preprocessor**: The data loader uses `AutoImageProcessor` from
  HuggingFace `transformers` (compatible with both CLIP and ViT-MAE models). -->

<!-- ### Downstream-only install (no GPU)

The prediction tasks (clustering, annotation) only need lightweight packages:

```bash
pip install scanpy leidenalg scikit-learn anndata pandas numpy
```

The `spatia` package exposes a single model class with swappable encoders:

```python
from spatia import SPATIA

model = SPATIA.from_config(
    n_genes=30000,
    gene_encoder_type="flash_transformer",   # or "standard_transformer"
    image_encoder_type="vitmae",             # or "clip" or None
    d_model=256, nlayers=8, nhead=4,
)
``` -->

---

## MIST Dataset Construction

MIST (Multimodal Imaging and Spatial Transcriptomics) is the pretraining
dataset. It combines cell-level gene expression with morphology image crops
from Xenium spatial transcriptomics data. Construction has three stages:

### Stage A: Crop cell images into LMDB

Crop cell-centered patches from Xenium morphology TIFF images and store them
in an LMDB database.

**Input layout** (per Xenium dataset):
```
dataset_dir/
├── morphology_mip.ome.tif    # or morphology.ome.tif / DAPI.tif
├── cells.parquet             # cell centroids + boundaries
└── cell_feature_matrix.h5    # (optional, for building h5ad)
```

**Crop pipeline:**
```bash
cd gene_encoders/SPATIA-scprint

# Standard Xenium format (cells.parquet + morphology.ome.tif)
python scripts/0510_crop_images_cell_refactored.py \
    --output-lmdb /path/to/output/dataset_name.lmdb \
    --output-size 256 \
    --cache /path/to/cache

# SPATCH format (adata.h5ad + DAPI.tif, for COAD/HCC/OV datasets)
python scripts/0510_crop_images_cell_spatch.py \
    --input-dir /path/to/SPATCH/Xenium-5K \
    --output-lmdb /path/to/output/lmdb \
    --dataset-name HCC
```

**Image processing details:**
- TIFF max intensity projection across channels
- Normalize to uint8 (0-255)
- Crop around cell centroid (adaptive size from cell boundaries, or default 32px radius)
- Resize to 256x256
- Store as raw bytes in LMDB with key format: `{dataset_name}/{cell_id}`
- Coordinate mapping: `pixel_x = spatial_y / 0.2125`, `pixel_y = spatial_x / 0.2125` (Xenium coordinate swap)

### Stage B: Build annotated h5ad and register in lamindb

Annotate each dataset with ontology metadata and add to a lamindb Collection:

```bash
cd gene_encoders/SPATIA-scprint

python scripts/0512_add_single_dataset.py \
    /path/to/adata.h5ad \
    --tissue lung --disease normal \
    --dataset_name xenium_lung \
    --collection_name xenium_all_0212
```

Required metadata columns (added automatically by the script):
- `organism_ontology_term_id` (e.g., `NCBITaxon:9606`)
- `cell_type_ontology_term_id`, `tissue_ontology_term_id`, `disease_ontology_term_id`
- `assay_ontology_term_id`, `sex_ontology_term_id`, `development_stage_ontology_term_id`
- `donor_id`, `dataset_name`, `index` (cell ID matching LMDB keys)

### Stage C: Merge per-dataset LMDBs (optional)

Consolidate multiple per-dataset LMDBs into a single file:

```bash
python scripts/0514_merge_lmdb.py \
    --input-dir /path/to/per_dataset_lmdbs/ \
    --output /path/to/merged/all.lmdb
```

### Data loading at training time

The training data loader (`scdataloader.data_spatial.Dataset`) reads:
1. Gene expression from a lamindb Collection (multiple h5ad files)
2. Cell images from LMDB files (supports multiple scales)

LMDB environments are mapped to image keys in the batch:
- 1st LMDB path -> `image` (cell-level crop)
- 2nd LMDB path -> `region_image` (niche-level, optional)
- 3rd LMDB path -> `tissue_image` (tissue-level, optional)

Images are preprocessed at load time: 256x256 grayscale -> stack to RGB ->
`AutoImageProcessor` (ViT-MAE) -> `(3, 224, 224)` float tensor.

---


## Representation Training (Stage 1)

### Prerequisites

Before training, you need:
1. **MIST dataset**: constructed via the pipeline above (lamindb Collection + LMDB)
2. **Base scPRINT checkpoint**: download with `cd gene_encoders/SPATIA-scprint/data/model && bash download.sh`
3. **Gene embeddings**: `data/generated/gene_embeddings.parquet` (from the scPRINT pretrained model)
4. **Biomart gene positions**: `data/main/biomart_pos.parquet` (shipped in this repo)

### Configure paths

Edit `gene_encoders/SPATIA-scprint/config/0510_base_spatial_all_crossattention_medium.yml`:

```yaml
data:
  spatial_datadir: /your/path/to/lmdb/all.lmdb     # <- LMDB from Stage C
  collection_name: xenium_all_0212                   # <- lamindb Collection from Stage B
  gene_embeddings: ./data/generated/gene_embeddings.parquet
  do_gene_pos: ./data/main/biomart_pos.parquet
model:
  ckpt_path: ./data/model/scPRINT/medium.ckpt       # <- base scPRINT checkpoint
  clip_model_type: "facebook/vit-mae-base"           # ViT-MAE image encoder
```

### Run training

```bash
# SPATIA-scprint (primary, requires lamindb + LMDB)
cd gene_encoders/SPATIA-scprint
scprint_spatial fit --config config/0510_base_spatial_all_crossattention_medium.yml \
    --model.ckpt_path data/model/scPRINT/medium.ckpt \
    --data.batch_size 8

# SPATIA-scgpt (alternative, uses accelerate for multi-GPU)
cd gene_encoders/SPATIA-scgpt
bash scgpt_spatial/0704_train_spatial_4h100.sh
```

Training uses PyTorch Lightning with bf16 mixed precision, DDP strategy,
and Weights & Biases logging.

---

## Prediction Tasks

Clustering and cell-annotation benchmarks (Tables 2 & 4). Runs on frozen embeddings — no GPU needed once embeddings are extracted.

See **[prediction_tasks/README.md](prediction_tasks/README.md)** for the full protocol: embedding extraction, dataset setup, resolution sweeps, and evaluation scripts.

---

## Generation Task

OT-based data pairing (Stage 2) followed by flow-matching image generation (Stage 3).

See **[generative_tasks/spatia_flow/README.md](generative_tasks/spatia_flow/README.md)** for training config, data pairing, and inference details.

---

<!-- ## Model Architecture

Cell level fuses a ViT-MAE image branch with a transformer gene branch via
cross-attention (visual tokens as queries, gene tokens as keys/values). Niche
and tissue levels aggregate cell embeddings with positional context. See the
paper for the full hierarchy.

| Component | Value |
|-----------|-------|
| Gene encoder | Flash Transformer, d_model 256, 8 layers |
| Image encoder | ViT-MAE base (12 enc / 8 dec layers, patch 16) |
| Fusion | Cross-attention (gene keys/values, image queries) |
| Image init | ImageNet-1k pretrained (`facebook/vit-mae-base`) |
| Gene init | scPRINT pretrained (mandatory) |
| Total params | ~155M (gene encoder ~40M + ViT-MAE ~86M + fusion + decoders) |
| Training precision | bf16-mixed |

--- -->

## Citation

```bibtex
@article{kong2025spatia,
  title={Spatia: Multimodal model for prediction and generation of spatial cell phenotypes},
  author={Kong, Zhenglun and Qiu, Mufan and Boesen, John and Lin, Xiang and Yun, Sukwon and Chen, Tianlong and Kellis, Manolis and Zitnik, Marinka},
  journal={ArXiv},
  pages={arXiv--2507},
  year={2025}
}
```

## Contact

- Zhenglun Kong — [zhenglun_kong@hms.harvard.edu](mailto:zhenglun_kong@hms.harvard.edu)
- Marinka Zitnik — [marinka@hms.harvard.edu](mailto:marinka@hms.harvard.edu)

Zitnik Lab, Harvard Medical School
