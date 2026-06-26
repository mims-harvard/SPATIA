#!/bin/bash
# Adjust these paths before running

set -eo pipefail

source ~/.bashrc
conda activate spatia

export NUMBA_CACHE_DIR=/tmp/numba_cache_$USER
mkdir -p $NUMBA_CACHE_DIR

REPO=/path/to/SPATIA
PUB=$REPO
NSDIR=/path/to/spatial_baselines
CKPT=/path/to/spatia-scgpt/checkpoint

echo "=============================================="
echo "CosMx Table 2 Pipeline"
echo "Node: $(hostname)  |  Start: $(date)"
echo "=============================================="

# -- Step 0: Unzip transcriptome --
echo ""
echo "[Step 0] Unzipping CosMx transcriptome..."
COSMX=$NSDIR/SPATCH/CosMx-6K/HCC
if [ ! -f "$COSMX/adata.h5ad" ]; then
    unzip -o $COSMX/transcriptome.zip -d $COSMX/
    # Flatten: transcriptome/adata.h5ad -> adata.h5ad
    if [ -f "$COSMX/transcriptome/adata.h5ad" ]; then
        mv -f $COSMX/transcriptome/adata.h5ad $COSMX/adata.h5ad
        echo "  Moved transcriptome/adata.h5ad -> adata.h5ad"
    fi
fi
echo "  adata.h5ad: $(ls -lh $COSMX/adata.h5ad 2>/dev/null | awk '{print $5}')"

# Quick sanity check
python -c "
import scanpy as sc
a = sc.read_h5ad('$COSMX/adata.h5ad')
print(f'  CosMx HCC: {a.n_obs} cells x {a.n_vars} genes')
print(f'  spatial: {\"spatial\" in a.obsm}')
print(f'  annotation: {\"annotation\" in a.obs.columns}')
ann = a.obs['annotation']
print(f'  valid annotations: {ann.notna().sum()}/{len(ann)}')
if 'DAPI resolution' in a.uns:
    print(f'  DAPI resolution: {a.uns[\"DAPI resolution\"]}')
else:
    print('  WARNING: no DAPI resolution in uns, will use default')
"

# -- Step 1: Build LMDB (cell crops) --
echo ""
echo "[Step 1] Building CosMx cell LMDB..."
LMDB_DIR=$NSDIR/spatia_t2/lmdb/cell_spatch_CosMx-6K
if [ -d "$LMDB_DIR/HCC.lmdb" ] && [ "$(ls -A $LMDB_DIR/HCC.lmdb 2>/dev/null)" ]; then
    echo "  LMDB already exists at $LMDB_DIR/HCC.lmdb, skipping"
else
    python $PUB/gene_encoders/SPATIA-scprint/scripts/0510_crop_images_cell_spatch.py \
        --image-file DAPI.tif --output-size 256 \
        --input-dir $NSDIR/SPATCH/CosMx-6K \
        --output-lmdb $LMDB_DIR \
        --cache $NSDIR/spatia_t2/cache \
        --dataset-name HCC
fi

# -- Step 2: Extract fused multimodal embeddings (GPU) --
echo ""
echo "[Step 2] Extracting CosMx fused embeddings (GPU)..."
EMB_DIR=$NSDIR/spatia_t2/embeddings
cd $PUB/gene_encoders/SPATIA-scgpt/tutorials
python extract_multimodal_embeddings.py \
    --spatial-config-path $CKPT/config.json \
    --spatial-weight-path $CKPT/best_model.pt \
    --h5ad-file $COSMX/adata.h5ad \
    --spatial-datadir $LMDB_DIR/HCC.lmdb \
    --output-path $NSDIR/spatia_t2/emb_cosmx \
    --gene-col index \
    --batch-size 64 --num-workers 8 \
    --platform CosMx --table2-emb-dir $EMB_DIR

# -- Step 3: PCA baseline for CosMx --
echo ""
echo "[Step 3] Computing PCA baseline for CosMx..."
python -c "
import scanpy as sc, numpy as np
from pathlib import Path

adata = sc.read_h5ad('$COSMX/adata.h5ad')
print(f'  CosMx: {adata.n_obs} cells x {adata.n_vars} genes')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), flavor='seurat_v3' if adata.n_vars >= 2000 else 'seurat')
sc.tl.pca(adata, n_comps=50)
emb = adata.obsm['X_pca']
out = Path('$EMB_DIR/pca')
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'CosMx_embeddings.npy', emb)
print(f'  PCA embedding: {emb.shape} -> {out / \"CosMx_embeddings.npy\"}')
"

echo ""
echo "=============================================="
echo "CosMx pipeline DONE at $(date)"
echo "=============================================="
echo ""
echo "Embeddings saved to:"
echo "  spatia: $EMB_DIR/spatia/CosMx_embeddings.npy"
echo "  pca:    $EMB_DIR/pca/CosMx_embeddings.npy"
