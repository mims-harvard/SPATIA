#!/bin/bash
# Example script for extracting SPATIA embeddings
# Modify the paths below according to your setup

# ============================================
# USER: Set these paths before running
# ============================================
# REPO       - Path to the SPATIA repository root
# CKPT       - Path to SPATIA-scGPT model checkpoint directory
# DATA_DIR   - Path to your datasets directory
# ============================================

eval "$(conda shell.bash hook)"
conda activate spatia

set -euo pipefail

# Configuration
MODEL_DIR="${CKPT:-/path/to/spatia-scgpt/checkpoint}"
H5AD_FILE="${DATA_DIR:-/path/to/data}/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/adata_processed_0212_with_cell_state.h5ad"
OUTPUT_DIR="${REPO:-/path/to/SPATIA}/embeddings"

# Example 1: Extract single embedding (uses best_model.pt)
echo "Example 1: Single embedding extraction"
python extract_embedding_simple.py --h5ad "$H5AD_FILE" --model "$MODEL_DIR" --step 0 --seed 0 --output "${OUTPUT_DIR}/SPATIA_embedding.npy"

# # Example 2: Batch extraction (multiple steps and seeds)
# echo ""
# echo "Example 2: Batch extraction"
# python extract_embedding_simple.py --h5ad "$H5AD_FILE" --model "$MODEL_DIR" --steps 10000 20000 30000 --seeds 0 1 2 --output "$OUTPUT_DIR"

echo ""
echo "Done! Check output: $OUTPUT_DIR"
