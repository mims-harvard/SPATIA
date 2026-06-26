# Installation

**Requirements:** Python 3.10, CUDA 12.1+, GPU with 40 GB+ VRAM (A100/H100 recommended).

## Conda Environment

The simplest way is to create the environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate spatia
```

Or build manually:

```bash
conda create -n spatia python=3.10
conda activate spatia

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

## Package Installation

**Install in this order** — `data_processing` must come before `SPATIA-scprint`.

```bash
# Flash Attention (required for the scPRINT transformer)
pip install flash-attn --no-build-isolation

# Data processing package
pip install -e data_processing

# SPATIA-scprint representation model
pip install -e gene_encoders/SPATIA-scprint
```

## Generation Dependencies

For Stage 3 (flow-matching image generation), install extra packages:

```bash
pip install torchdiffeq flow-matching torch-fidelity
```

## Notes

**numba cache** — On shared filesystems with read-only site-packages, export a writable cache before running anything that imports `scanpy`:

```bash
export NUMBA_CACHE_DIR=/tmp/numba_cache_$USER && mkdir -p $NUMBA_CACHE_DIR
```

**Flash Attention** — Requires CUDA 11.6+ and GLIBC 2.17+. If the build fails, the SPATIA-scgpt path has a built-in fallback to standard `nn.MultiheadAttention`.

**lamindb** (pretraining only) — Required for building the MIST pretraining dataset. Not needed for downstream evaluation.

```bash
pip install "lamindb[bionty]==0.76.12"
lamin init --storage ./data --name scprint_db
python -c "import bionty as bt; bt.base.reset_sources()"
```
