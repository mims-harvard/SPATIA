#!/bin/bash
# Extract SPATIA embeddings for Xenium Breast Cancer dataset

# ============================================
# USER: Set these paths before running
# ============================================
# CKPT       - Path to SPATIA-scGPT model checkpoint directory
# DATA_DIR   - Path to your datasets directory
# REPO       - Path to the SPATIA repository root
# ============================================

eval "$(conda shell.bash hook)"
conda activate spatia

set -euo pipefail

# Configuration
MODEL_DIR="${CKPT:-/path/to/spatia-scgpt/checkpoint}"
H5AD_ORIGINAL="${DATA_DIR:-/path/to/data}/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/adata_processed_0212_with_cell_state.h5ad"
H5AD_FIXED="${DATA_DIR:-/path/to/data}/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/adata_processed_0212_with_cell_state_fixed.h5ad"
OUTPUT_DIR="${REPO:-/path/to/SPATIA}/embedding"

# Check if fixed dataset exists, if not create it
if [ ! -f "$H5AD_FIXED" ]; then
    echo "Fixing gene names in Xenium dataset..."
    python fix_xenium_genes.py
fi

echo "Extracting SPATIA embeddings for Xenium Breast Cancer dataset..."
python extract_embedding_simple.py \
    --h5ad "$H5AD_FIXED" \
    --model "$MODEL_DIR" \
    --step 0 \
    --seed 0 \
    --output "${OUTPUT_DIR}/SPATIA_embedding_xenium.npy"

echo ""
echo "Done! Embedding saved to: ${OUTPUT_DIR}/SPATIA_embedding_xenium.npy"
echo "Now you can run: bash run_visualize.sh"
