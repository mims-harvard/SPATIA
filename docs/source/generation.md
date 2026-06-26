# Generation Pipeline (Stages 2 & 3)

State-conditioned cell image generation: given a control cell H&E crop, generate the predicted target cell image after a biological perturbation. The pipeline has two stages:

1. **Stage 2** — OT-based perturbation pair construction (`generative_tasks/data_pairing_for_FM/`)
2. **Stage 3** — Flow-matching image generation (`generative_tasks/spatia_flow/`)

```{image} ../../static/images/visual.png
:width: 600px
:align: center
```

---

## Stage 2: OT-Based Perturbation Pairing

Five biological transitions are modeled across two tasks:

**Task 1 — Tumor Progression:**

| Transition | Control → Target |
|------------|-----------------|
| EMT | Epi_FOXA1+ → EMT-Epi1_CEACAM6+ |
| Proliferation | Epi_FOXA1+ → Epi_CENPF+ |
| Lineage conversion | Epi_FOXA1+ → mgEpi_KRT14+ |

**Task 2 — Immune Infiltration:**

| Transition | Control → Target |
|------------|-----------------|
| T-cell activation | tcm_CD4+T → eff_CD8+T1 |
| Angiogenesis | EC_CAVIN2+ → EC_CLEC14A+ |

### Cell-level pairing

```bash
cd generative_tasks/data_pairing_for_FM

python generate_spatia_pairs.py \
    --adata /path/to/adata_with_cell_states.h5ad \
    --out_dir ./spatia_pairs_output \
    --lmdb_path /path/to/xenium_he_rep1_192px.lmdb \
    --state_col cell_states \
    --niche_col niche
```

Or use the convenience script:

```bash
bash run_spatia_pairing.sh
```

**Outputs:**

```
spatia_pairs_output/
├── perturbation_pairs.csv      # OT-matched pairs with transition metadata
├── delta_g_signatures.npz      # Per-transition Δg vectors (mean expression shift)
├── delta_m_signatures.npz      # Per-transition Δm vectors (morphology shift)
└── pairing_config.json
```

### Niche-level pairing (grid-based)

```bash
python generate_grid_niche_pairs.py \
    --niche_adata /path/to/grid_niche_256_256/DATASET.h5ad \
    --cell_adata /path/to/cell_level/DATASET_cells.h5ad \
    --out_dir ./niche_pairs_output \
    --state_col cell_states \
    --grid_size 256 \
    --min_cells 5 --min_fraction 0.05 \
    --include_ot_confidence
```

---

## Stage 3: Flow-Matching Image Generation

### Pretrained Checkpoints

Download CellFlux pretrained checkpoints from [HuggingFace](https://huggingface.co/Perturbation/CellFlux) and place them in `generative_tasks/spatia_flow/pretrained_checkpoints/`.

### Configure data paths

Edit `generative_tasks/spatia_flow/configs/spatia_bio.yaml`:

```yaml
adata_path:  /path/to/adata_with_cell_states.h5ad
pairs_csv:   /path/to/spatia_pairs_output/perturbation_pairs.csv
lmdb_path:   /path/to/xenium_he_rep1_192px.lmdb
delta_g_npz: /path/to/spatia_pairs_output/delta_g_signatures.npz
delta_m_npz: /path/to/spatia_pairs_output/delta_m_signatures.npz

spatia_model_path: /path/to/spatia-scgpt/checkpoint/
spatia_vocab_path: /path/to/spatia-scgpt/checkpoint/vocab.json
spatia_gene_stats: /path/to/spatia-scgpt/checkpoint/all_dict_mean_std.csv
```

### Training

```bash
cd generative_tasks/spatia_flow

# Quick test (single GPU, ~1 min)
python train_xenium_spatia.py \
    --config spatia_bio \
    --batch_size 4 --epochs 2 --test_run \
    --output_dir /tmp/spatia_test_run

# Full training (multi-GPU, ~28h for 100 epochs on 2×H100)
python -m torch.distributed.run --nproc_per_node=2 --master_port=29502 \
    train_xenium_spatia.py \
    --config spatia_bio \
    --batch_size 8 --accum_iter 4 --epochs 100 \
    --output_dir ./outputs/my_experiment \
    --eval_frequency 10

# Resume from checkpoint
python -m torch.distributed.run --nproc_per_node=2 --master_port=29502 \
    train_xenium_spatia.py \
    --config spatia_bio \
    --batch_size 8 --epochs 100 \
    --output_dir ./outputs/my_experiment \
    --resume ./outputs/my_experiment/checkpoint.pth
```

:::{note}
Use `python -m torch.distributed.run` instead of `torchrun` to ensure the conda environment's Python is used.
:::

### Output structure

```
outputs/my_experiment/
├── args.json                      # Full config snapshot
├── checkpoint.pth                 # Latest checkpoint (supports resume)
├── checkpoint-{epoch}.pth         # Periodic checkpoints
├── log.txt                        # Per-epoch loss (JSON lines)
└── snapshots/{epoch}_0.png        # Generated image grids (ctrl → generated → target)
```

### Evaluation (FID)

```bash
python train_xenium_spatia.py \
    --config spatia_bio \
    --eval_only \
    --resume ./outputs/my_experiment/checkpoint.pth \
    --output_dir ./outputs/eval \
    --fid_samples 200
```

### Morphology proxy encoder

The morph proxy encoder (`checkpoints/morph_proxy_encoder.pth`) is pretrained on cell regionprops features and provides a morphological reference for the conditioning signal. It is included in the repository and loaded automatically via the config key `morph_proxy_checkpoint`.

To retrain it:

```bash
python training/train_morph_proxy.py \
    --lmdb_path /path/to/data/xenium_he_rep1_192px.lmdb \
    --output_dir ./checkpoints
```

---

## Niche-Level Generation

```{image} ../../static/images/levels_niche.png
:width: 600px
:align: center
```

Use niche-level pairs (from `generate_grid_niche_pairs.py`) with the niche dataset class:

```python
from spatia_niche_dataset import SpatiaNicheDataset, create_niche_dataloaders

dataset = SpatiaNicheDataset(
    adata_path="/path/to/adata.h5ad",
    pairs_csv="niche_pairs_output/niche_pairs.csv",
    delta_g_npz="niche_pairs_output/niche_delta_g.npz",
    wsi_image_path="/path/to/he_image.tif",
    image_size=(512, 512),
)

train_loader, val_loader = create_niche_dataloaders(dataset, batch_size=8)
```
