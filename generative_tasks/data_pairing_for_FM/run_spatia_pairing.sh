#!/bin/bash

# =============================================================================
# SPATIA Perturbation Pairing Pipeline
# =============================================================================
# This script runs the OT-based perturbation pairing for SPATIA.
# 
# Usage:
#   bash run_spatia_pairing.sh              # Run with defaults
#   sbatch run_spatia_pairing.sh            # Submit as SLURM job
# =============================================================================

set -e  # Exit on error

# Configuration - Modify these paths as needed
# ============================================
# USER: Set DATA_DIR to your datasets root before running
# ============================================
DATA_DIR="${DATA_DIR:-/path/to/data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADATA_PATH="${DATA_DIR}/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/adata_processed_0212_with_cell_state.h5ad"
OUTPUT_DIR="${SCRIPT_DIR}/spatia_pairs_output"
STATE_COL="cell_states"
NICHE_COL="niche"

# LMDB path for morphology extraction (set to empty string to disable)
# When provided, extracts 10-dim morphology features and computes delta_m signatures
LMDB_PATH="${DATA_DIR}/lmdb/xenium_he_0919.lmdb"

# Print configuration
echo "=============================================="
echo "SPATIA Perturbation Pairing Pipeline"
echo "=============================================="
echo "Timestamp: $(date)"
echo "Script directory: ${SCRIPT_DIR}"
echo "AnnData path: ${ADATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "State column: ${STATE_COL}"
echo "Niche column: ${NICHE_COL}"
if [ -n "${LMDB_PATH}" ]; then
    echo "LMDB path: ${LMDB_PATH} (morphology extraction ENABLED)"
else
    echo "LMDB path: Not set (morphology extraction DISABLED)"
fi
echo "=============================================="

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    source ~/.bashrc 2>/dev/null || true
    conda activate spatia 2>/dev/null || {
        echo "Warning: Could not activate spatia environment"
        echo "Continuing with current environment..."
    }
fi

# Check Python environment
echo ""
echo "Python environment:"
which python
python --version

# Check required packages
echo ""
echo "Checking required packages..."
python -c "import scanpy, torch, numpy, pandas; print('Core packages found')" || {
    echo "Error: Missing required packages. Please install: scanpy, torch, numpy, pandas"
    exit 1
}

# Check morphology extraction packages (only if LMDB_PATH is set)
if [ -n "${LMDB_PATH}" ]; then
    python -c "import cv2, skimage, lmdb, PIL; print('Morphology extraction packages found')" || {
        echo "Warning: Missing packages for morphology extraction (cv2, skimage, lmdb, PIL)"
        echo "Morphology extraction will be disabled."
        LMDB_PATH=""
    }
fi

# Check input file exists
if [ ! -f "${ADATA_PATH}" ]; then
    echo "Error: AnnData file not found: ${ADATA_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the pairing script
echo ""
echo "Running perturbation pairing..."
echo "=============================================="

cd "${SCRIPT_DIR}"

# Build command with optional LMDB path
CMD="python generate_spatia_pairs.py \
    --adata ${ADATA_PATH} \
    --out_dir ${OUTPUT_DIR} \
    --state_col ${STATE_COL} \
    --niche_col ${NICHE_COL} \
    --pca_dim 50 \
    --sinkhorn_eps 0.05 \
    --sinkhorn_iter 200 \
    --min_cells 5 \
    --max_cells 500 \
    --seed 42"

# Add LMDB path if specified (enables morphology extraction)
if [ -n "${LMDB_PATH}" ]; then
    CMD="${CMD} --lmdb_path ${LMDB_PATH}"
fi

# Execute
eval ${CMD}

echo ""
echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo ""
echo "Output files:"
ls -la "${OUTPUT_DIR}/"
echo ""
echo "To use the generated data in SPATIA-CellFlux config:"
echo "  pairs_csv: '${OUTPUT_DIR}/perturbation_pairs.csv'"
echo "  delta_g_npz: '${OUTPUT_DIR}/delta_g_signatures.npz'"
if [ -n "${LMDB_PATH}" ]; then
    echo "  delta_m_npz: '${OUTPUT_DIR}/delta_m_signatures.npz'"
    echo ""
    echo "Note: pairs_csv now includes 'ot_confidence' column for weighted training"
fi
