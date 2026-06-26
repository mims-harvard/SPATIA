#!/bin/bash
# Adjust these paths before running

set -eo pipefail

source ~/.bashrc
conda activate spatia

export NUMBA_CACHE_DIR=/tmp/numba_cache_$USER
mkdir -p $NUMBA_CACHE_DIR

REPO=/path/to/SPATIA
NSDIR=/path/to/spatial_baselines
EMB_DIR=$NSDIR/spatia_t2/embeddings
EVAL_DIR=$NSDIR/spatia_t2/eval_cosmx
RESULTS_DIR=$NSDIR/spatia_t2/results_cosmx

mkdir -p $EVAL_DIR $RESULTS_DIR

echo "=============================================="
echo "TABLE 2: CosMx clustering"
echo "Node: $(hostname)  |  Start: $(date)"
echo "=============================================="

# Prepare CosMx eval data
python -c "
import scanpy as sc, numpy as np
from pathlib import Path

adata = sc.read_h5ad('$NSDIR/SPATCH/CosMx-6K/HCC/adata.h5ad')
valid = adata.obs['annotation'].notna()
print(f'Valid cells: {valid.sum()}/{len(adata)}')

adata_valid = adata[valid].copy()
adata_valid.write_h5ad(Path('$EVAL_DIR/CosMx_10K.h5ad'))
print(f'Saved CosMx_10K.h5ad: {adata_valid.shape}')

n_valid = int(valid.sum())
for model in ['spatia', 'pca']:
    emb_path = Path('$EMB_DIR') / model / 'CosMx_embeddings.npy'
    if emb_path.exists():
        emb = np.load(emb_path)
        if len(emb) == len(adata):
            emb_filt = emb[valid.values]
            np.save(emb_path, emb_filt)
            print(f'  {model}: filtered {len(emb)} -> {len(emb_filt)}')
        elif len(emb) == n_valid:
            print(f'  {model}: already filtered ({len(emb)})')
        else:
            print(f'  {model}: size {len(emb)} (total={len(adata)}, valid={n_valid})')
"

echo ""
echo "Running clustering..."
cd $REPO/prediction_tasks/scripts
python multi_seed_clustering.py \
  --task table2 \
  --table2_emb_dir $EMB_DIR \
  --table2_data_dir $EVAL_DIR \
  --output_dir $RESULTS_DIR \
  --n_seeds 5 \
  --table2_models spatia pca

echo ""
echo "DONE at $(date)"
