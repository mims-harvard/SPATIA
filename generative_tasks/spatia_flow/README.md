# SPATIA-CellFlux

State-conditioned cell image generation via flow matching. Given a control cell H&E crop, generates the predicted target cell image after a biological perturbation.

## Pretrained Checkpoints

Download CellFlux pretrained checkpoints from [HuggingFace](https://huggingface.co/Perturbation/CellFlux) and place them in the `pretrained_checkpoints/` directory before training.

## 1. Environment Setup

```bash
conda activate spatia

# Verify dependencies
python -c "
import torch, torchvision, flow_matching, torchdiffeq
import scanpy, anndata, lmdb, cv2, skimage, PIL
print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')
"

# Install if missing
pip install torchdiffeq flow-matching torch-fidelity
```

## 2. Data Preparation

Build LMDB + OT pairs from raw Xenium data (one-time, ~65 min):

### Step 1: Crop H&E images → LMDB

```bash
cd gene_encoders/SPATIA-scprint

# Standard Xenium format
python scripts/0510_crop_images_cell_refactored.py \
    --output-lmdb /path/to/output/xenium_he_rep1_192px.lmdb \
    --output-size 256 \
    --cache /path/to/cache
```

Output: `xenium_he_rep1_192px.lmdb` — cell-level H&E crops (256×256)

### Step 2: OT-based perturbation pairing

```bash
cd generative_tasks/data_pairing_for_FM

# Edit DATA_DIR and ADATA_PATH in run_spatia_pairing.sh, then:
bash run_spatia_pairing.sh
```

Or run directly:

```bash
python generate_spatia_pairs.py \
    --adata /path/to/adata_with_cell_states.h5ad \
    --out_dir ./spatia_pairs_output \
    --lmdb_path /path/to/xenium_he_rep1_192px.lmdb \
    --state_col cell_states \
    --niche_col niche
```

Output files:
- `spatia_pairs_output/perturbation_pairs.csv` — OT pairs across biological transitions
- `spatia_pairs_output/delta_g_signatures.npz` — per-transition gene shift (n_transitions × n_genes)
- `spatia_pairs_output/delta_m_signatures.npz` — per-transition morphology shift (n_transitions × 10)

See `generative_tasks/data_pairing_for_FM/README.md` for niche-level pairing and full parameter reference.

## 3. Training

Edit `configs/spatia_bio.yaml` to set your data paths:

```yaml
adata_path: /path/to/adata_with_cell_states.h5ad
pairs_csv:  /path/to/spatia_pairs_output/perturbation_pairs.csv
lmdb_path:  /path/to/xenium_he_rep1_192px.lmdb
delta_g_npz: /path/to/spatia_pairs_output/delta_g_signatures.npz
delta_m_npz: /path/to/spatia_pairs_output/delta_m_signatures.npz

spatia_model_path: /path/to/spatia-scgpt/checkpoint/
spatia_vocab_path: /path/to/spatia-scgpt/checkpoint/vocab.json
spatia_gene_stats: /path/to/spatia-scgpt/checkpoint/all_dict_mean_std.csv
```

Then run:

```bash
cd SPATIA/generative_tasks/spatia_flow

# Test run (single GPU, ~1 min)
python train_xenium_spatia.py \
    --config spatia_bio \
    --batch_size 4 --epochs 2 --test_run \
    --output_dir /tmp/spatia_test_run

# Full training (multi-GPU DDP, ~28h for 100 epochs on 2×H100)
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
    --batch_size 8 --accum_iter 4 --epochs 100 \
    --output_dir ./outputs/my_experiment \
    --eval_frequency 10 \
    --resume ./outputs/my_experiment/checkpoint.pth
```

> Use `python -m torch.distributed.run` instead of `torchrun` to ensure the conda env Python is used.


## 4. Output Structure & Visualization

Training snapshots are auto-saved every `eval_frequency` epochs:

```
outputs/my_experiment/
├── args.json                  # Full config snapshot
├── checkpoint.pth             # Latest checkpoint (supports resume)
├── checkpoint-{epoch}.pth     # Periodic checkpoints
├── log.txt                    # Per-epoch loss (JSON lines)
├── snapshots/{epoch}_0.png    # Generated image grids (ctrl → generated → target)
└── train_stdout.log           # Full training log
```

Evaluate with FID:

```bash
python train_xenium_spatia.py \
    --config spatia_bio \
    --eval_only \
    --resume ./outputs/my_experiment/checkpoint.pth \
    --output_dir ./outputs/eval \
    --fid_samples 200
```

