# SPATIA Perturbation Pairing Pipeline

This directory contains the code for generating **optimal transport-based perturbation pairs** for training the SPATIA flow matching model, as described in Appendix D of the SPATIA paper.

## Pairing Modes

This folder supports **cell-level pairing** and **niche-level pairing**.
For niche-level pairing, there are two implementations: **grid-based (primary)** and **region-based (optional)**.

| Mode | Script | Pairs | Image Source | Typical Patch Size | Use Case |
|------|--------|-------|--------------|--------------------|----------|
| **Cell-level** | `generate_spatia_pairs.py` | Individual cells | Cell crops from LMDB (optional) | ~128×128 | Single-cell morphology + expression |
| **Niche-level (Grid-based, primary)** | `generate_grid_niche_pairs.py` | Fixed grid patches | Precomputed grid LMDB (external) | 256×256 | Microenvironment dynamics on a fixed grid |
| **Niche-level (Region-based, optional)** | `generate_niche_pairs.py` | Variable spatial regions | WSI crops or LMDB (optional) | ~512×512 | Microenvironment dynamics with variable regions |

## Quick Start

### Niche-Level Pairing (Grid-based, Primary)

This is the workflow used by `run_niche_pairing.sh` (Option 1: precomputed grid niche data).

```bash
# Run grid-based niche pairing (precomputed grid_niche_* inputs)
bash run_niche_pairing.sh

# Or run directly
python generate_grid_niche_pairs.py \
    --niche_adata /path/to/grid_niche_256_256/DATASET.h5ad \
    --cell_adata /path/to/cell_level/DATASET_cells.h5ad \
    --out_dir ./niche_pairs_output \
    --state_col cell_states \
    --cell_id_col index \
    --grid_size 256 \
    --min_cells 5 \
    --min_fraction 0.05 \
    --max_pairs 200
```

Notes:
- `generate_grid_niche_pairs.py` does **not** require an LMDB path to *generate pairs*; LMDB is used later for visualization/training.
- Grid LMDB values may be stored as raw `uint8` arrays (often single-channel); visualization may apply a matplotlib colormap.

### Cell-Level Pairing

```bash
# Run cell-level pairing
bash run_spatia_pairing.sh

# Or run directly
python generate_spatia_pairs.py \
    --adata /path/to/adata.h5ad \
    --out_dir ./spatia_pairs_output \
    --state_col cell_states \
    --niche_col niche
```

### Niche-Level Pairing (Region-based, Optional)

```bash
# Run niche-level pairing
bash run_niche_pairing.sh

# Or run directly
python generate_niche_pairs.py \
    --adata /path/to/adata.h5ad \
    --out_dir ./niche_pairs_output \
    --state_col cell_states \
    --niche_col niche
```

## File Organization

### Primary Files (Use These)

| File | Description |
|------|-------------|
| `generate_spatia_pairs.py` | Cell-level OT pairing (+ optional morphology/Δm if LMDB provided) |
| `generate_grid_niche_pairs.py` | **Grid-based niche pairing (primary niche workflow)** |
| `generate_niche_pairs.py` | Region-based niche pairing (optional niche workflow) |
| `run_spatia_pairing.sh` | Run cell-level pipeline |
| `run_niche_pairing.sh` | Run niche-level pipeline (grid-based by default) |
| `visualize_niche_pairs.ipynb` | *(not included)* Visualize grid-niche outputs + LMDB patches; export paired images |
| `spatia_dataset.py` | PyTorch dataset for cell-level paired training |
| `spatia_niche_dataset.py` | Region-style niche dataset (expects `source_niche_id`/bbox columns) |

## Algorithm Overview

The pairing algorithm (Algorithm 1 in the paper) follows the same high-level template for cells and niches:

1. **For each biological transition** (e.g., Epi_FOXA1+ → EMT-Epi1_CEACAM6+)
2. **Select candidate source/target units**
    - Cell-level: optionally enforce same-niche constraints via `niche_col`
    - Grid-niche: select source grids enriched for `state_A` and target grids enriched for `state_B` (via `min_cells` / `min_fraction`)
    - Region-niche: analogous selection using region composition statistics
3. **OT pairing in expression space**
    - Extract expression vectors for source/target units
    - Project to PCA space (default `pca_dim=50` for cell-level; grid-niche uses the grid AnnData matrix)
    - Compute cost matrix $C_{ij}=\lVert g_i^A - g_j^B\rVert_2$
    - Solve entropy-regularized OT (Sinkhorn)
    - Extract hard assignment pairs from the transport plan
4. **Compute Δg per transition type** (Eq. 11)
   ```
   Δg_τ = (1/|P_τ|) Σ_{(i,j)∈P_τ} (g_j^t - g_i^c)
   ```
    This averages over all pairs of transition type τ.
5. **(Optional, cell-level)** compute Δm (morphology) signatures if an LMDB of cell crops is provided.

## Biological Transitions

The following transitions are defined in the paper:

### Task 1: Tumor Progression
| Transition | Control State | Target State |
|------------|---------------|--------------|
| EMT transition | Epi_FOXA1+ | EMT-Epi1_CEACAM6+ |
| Proliferation activation | Epi_FOXA1+ | Epi_CENPF+ |
| Lineage conversion | Epi_FOXA1+ | mgEpi_KRT14+ |

### Task 2: Immune Infiltration
| Transition | Control State | Target State |
|------------|---------------|--------------|
| T-cell activation | tcm_CD4+T | eff_CD8+T1 |
| Angiogenesis activation | EC_CAVIN2+ | EC_CLEC14A+ |

## Output Files

### Cell-Level outputs (`generate_spatia_pairs.py`)

Written to `--out_dir` (e.g. `./spatia_pairs_output/`):

```
spatia_pairs_output/
├── perturbation_pairs.csv      # Paired cell indices with metadata
├── delta_g_signatures.npz      # Per-transition Δg vectors
├── delta_m_signatures.npz      # (optional) Per-transition Δm vectors
└── pairing_config.json         # Configuration and statistics
```

### Grid-Niche outputs (`generate_grid_niche_pairs.py`)

Written to `--out_dir` (e.g. `./niche_pairs_output/`):

```
niche_pairs_output/
├── niche_pairs.csv             # Paired grid IDs with metadata (+ ot_confidence if --include_ot_confidence)
├── niche_delta_g.npz           # Per-transition Δg vectors
├── niche_delta_m.npz           # (optional, if --lmdb_path) Per-transition Δm vectors (morphology)
└── niche_config.json           # Configuration and statistics
```

### Cell-level: perturbation_pairs.csv columns
- `x_ctrl_id`: Index of control cell in AnnData
- `x_tgt_id`: Index of target cell in AnnData  
- `state_A`: Control state name
- `state_B`: Target state name
- `niche_ctrl`: Niche of control cell
- `niche_tgt`: Niche of target cell
- `transition_tag`: Biological transition type
- `task_name`: Task category (tumor_progression or immune_infiltration)

### delta_g_signatures.npz contents
- `delta_g_keys`: Array of transition tags (e.g., "EMT_transition")
- `delta_g_values`: Array of corresponding Δg vectors (shape: [n_genes])
- `genes`: Gene names matching the Δg vector dimensions

### Grid-niche: niche_pairs.csv columns

Grid-niche pairing is keyed by grid patch IDs and grid coordinates:
- `source_grid_id`, `target_grid_id` - Grid patch identifiers
- `source_grid_row`, `source_grid_col`, `target_grid_row`, `target_grid_col` - Grid coordinates
- `source_num_cells`, `target_num_cells` - Cell counts per grid
- `source_state_A_frac`, `target_state_B_frac` (and other state fraction columns)
- `state_A`, `state_B`, `transition_tag`, `task_name` - Transition metadata
- `ot_confidence` - (optional, with `--include_ot_confidence`) OT confidence score for weighted training

## Usage in Training

```python
from spatia_dataset import SpatiaPairedDataset, create_spatia_dataloaders

# Create dataset
dataset = SpatiaPairedDataset(
    adata_path="/path/to/adata.h5ad",
    pairs_csv="output/perturbation_pairs.csv",
    lmdb_path="/path/to/images.lmdb",
    delta_g_npz="output/delta_g_signatures.npz",
    return_expr=True,
    image_size=(256, 256)
)

# Create dataloaders
train_loader, val_loader = create_spatia_dataloaders(
    dataset, 
    batch_size=32,
    val_split=0.1
)

# Training loop
for batch in train_loader:
    ctrl_images = batch["ctrl_image"]      # [B, C, H, W]
    tgt_images = batch["tgt_image"]        # [B, C, H, W]
    ctrl_expr = batch["ctrl_expr"]         # [B, G]
    tgt_expr = batch["tgt_expr"]           # [B, G]
    delta_g = batch.get("delta_g")         # [B, G] - conditioning signal
    # ... training code
```

### Grid-niche / region-niche datasets

- Region-style niche training is supported by `spatia_niche_dataset.py`, but it expects region-style columns like `source_niche_id` and bounding boxes.
- Grid-niche outputs use `source_grid_id` / `target_grid_id` (and grid row/col). If you want to train directly on grid-niche pairs, you typically implement a small dataset that:
    - loads `niche_pairs.csv`
    - indexes pooled expressions from the grid AnnData by `grid_id`
    - loads images from the grid LMDB by key `<dataset_prefix>/grid_{row}_{col}_{grid_size}_{grid_size}`

## Configuration Options

### Cell-level (`generate_spatia_pairs.py`)

```
--adata           Path to AnnData h5ad file (required)
--out_dir         Output directory (required)
--lmdb_path       Optional LMDB path for morphology extraction
--state_col       Column for cell states (default: cell_states)
--niche_col       Column for spatial niche (default: niche)
--expr_layer      Expression layer in AnnData (default: X)
--pca_dim         PCA dimensions for OT (default: 50)
--sinkhorn_eps    Sinkhorn regularization (default: 0.05)
--sinkhorn_iter   Sinkhorn iterations (default: 200)
--min_cells       Min cells per state per niche (default: 5)
--max_cells       Max cells per group (default: 500)
--seed            Random seed (default: 42)
```

### Grid-niche (`generate_grid_niche_pairs.py`)

```
--niche_adata           Grid-niche AnnData (e.g. grid_niche_256_256/*.h5ad)
--cell_adata            Cell-level AnnData (for state info)
--out_dir               Output directory
--lmdb_path             Optional LMDB path for morphology extraction (Δm)
--dataset_prefix        Dataset prefix for LMDB keys (default: Xenium_FFPE_Human_Breast_Cancer_Rep1_outs)
--include_ot_confidence Include OT confidence scores in output CSV (flag)
--state_col             Column in cell_adata.obs containing cell states
--cell_id_col           Column in cell_adata.obs matching niche cell_ids (default: index)
--grid_size             Grid patch size in pixels (default: 256)
--min_cells             Min cells per grid for composition filtering
--min_fraction          Min fraction for source/target enrichment filtering
--max_pairs             Max pairs per transition
--seed                  Random seed
```

**New features (v2):**
- Optional `ot_confidence` column in `niche_pairs.csv` for weighted training (use `--include_ot_confidence`)
- Optional `niche_delta_m.npz` for morphology signatures (requires `--lmdb_path`)

**Backward compatibility:**
- Without `--lmdb_path` and `--include_ot_confidence`, output is identical to v1

## Niche-Level vs Cell-Level Details

### Cell-Level (`generate_spatia_pairs.py`)
- **Output**: `perturbation_pairs.csv` with `x_ctrl_id`, `x_tgt_id` (cell indices)
- **Images**: Single cell crops from LMDB (~128×128 pixels)
- **Expression**: Per-cell gene expression
- **Use with**: `spatia_dataset.py` → `SpatiaPairedDataset`

### Niche-Level (Grid-based, primary) (`generate_grid_niche_pairs.py`)
- **Output**: `niche_pairs.csv` with `source_grid_id`, `target_grid_id`, and grid row/col metadata
- **Images**: Precomputed grid patches stored in an external LMDB (used for visualization/training)
- **Expression**: Pooled expression per grid patch from the grid AnnData

### Niche-Level (Region-based, optional) (`generate_niche_pairs.py`)
- **Output**: `niche_pairs.csv` with `source_niche_id`, `target_niche_id` and region metadata (often includes bounding boxes)
- **Images**: Region crops from WSI or LMDB (resized to a fixed size for training)
- **Expression**: Pooled (mean) across cells in each region

### Region-niche Output Columns (typical)

```
source_niche_id, target_niche_id     # Niche identifiers
source_n_cells, target_n_cells       # Cell counts per niche
source_state_A_frac, source_state_B_frac  # Cell state proportions
target_state_A_frac, target_state_B_frac
source_bbox_xmin, source_bbox_ymin, source_bbox_xmax, source_bbox_ymax  # Bounding box (microns)
target_bbox_xmin, target_bbox_ymin, target_bbox_xmax, target_bbox_ymax
source_centroid_x, source_centroid_y  # Region centroids
target_centroid_x, target_centroid_y
state_A, state_B, transition_tag, task_name
```

### Using Niche-Level Data

```python
from spatia_niche_dataset import SpatiaNicheDataset, create_niche_dataloaders

dataset = SpatiaNicheDataset(
    adata_path="/path/to/adata.h5ad",
    pairs_csv="niche_pairs_output/niche_pairs.csv",
    delta_g_npz="niche_pairs_output/niche_delta_g.npz",
    wsi_image_path="/path/to/he_image.tif",  # For region cropping
    image_size=(512, 512),
)

train_loader, val_loader = create_niche_dataloaders(dataset, batch_size=8)
```

## Requirements

```
numpy
pandas
scanpy
torch
tqdm
```

## Citation

If you use this code, please cite the SPATIA paper.
