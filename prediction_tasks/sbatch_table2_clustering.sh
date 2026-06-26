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
EVAL_DIR=$NSDIR/spatia_t2/eval
RESULTS_DIR=$NSDIR/spatia_t2/results

rm -rf $EVAL_DIR $RESULTS_DIR
mkdir -p $EVAL_DIR $RESULTS_DIR

echo "=============================================="
echo "TABLE 2: Prepare eval data + run clustering"
echo "Node: $(hostname)  |  Start: $(date)"
echo "=============================================="

python -c "
import scanpy as sc, numpy as np
from pathlib import Path

adata = sc.read_h5ad('$NSDIR/SPATCH/Xenium-5K/HCC/adata.h5ad')
valid = adata.obs['annotation'].notna()
print(f'Valid cells: {valid.sum()}/{len(adata)}')

adata_valid = adata[valid].copy()
adata_valid.write_h5ad(Path('$EVAL_DIR/Xenium_10K.h5ad'))
print(f'Saved Xenium_10K.h5ad: {adata_valid.shape}')

n_valid = int(valid.sum())
for model in ['spatia', 'pca']:
    emb_path = Path('$EMB_DIR') / model / 'Xenium_embeddings.npy'
    if emb_path.exists():
        emb = np.load(emb_path)
        if len(emb) == len(adata):
            emb_filt = emb[valid.values]
            np.save(emb_path, emb_filt)
            print(f'  {model}: filtered {len(emb)} -> {len(emb_filt)}')
        elif len(emb) == n_valid:
            print(f'  {model}: already filtered ({len(emb)})')
        else:
            print(f'  {model}: UNEXPECTED size {len(emb)}')
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
cd -

echo ""
echo "DONE at $(date)"
