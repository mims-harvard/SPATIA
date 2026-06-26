#!/bin/bash

# =============================================================================
# SPATIA Niche-Level Perturbation Pairing Pipeline
# =============================================================================
# This script generates region/niche level pairs (vs cell-level pairs)
# Now includes:
#   - OT confidence scores for weighted training
#   - Delta_m morphology signatures (if LMDB provided)
# 
# Usage:
#   bash run_niche_pairing.sh              # Run with defaults
#   sbatch run_niche_pairing.sh            # Submit as SLURM job
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================
# USER: Set DATA_DIR to your datasets root before running
# ============================================
DATA_DIR="${DATA_DIR:-/path/to/data}"

# ============================================
# Option 1: Use pre-computed grid niche data (256x256 patches)
# This data already has pooled expression per grid patch
# ============================================
NICHE_ADATA_PATH="${DATA_DIR}/lmdb/grid_niche_256_256/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.h5ad"
NICHE_LMDB_PATH="${DATA_DIR}/lmdb/grid_niche_256_256/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.lmdb"
DATASET_PREFIX="Xenium_FFPE_Human_Breast_Cancer_Rep1_outs"

# ============================================
# Option 2: Use cell-level data and compute niche pooling
# ============================================
CELL_ADATA_PATH="${DATA_DIR}/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/adata_processed_0212_with_cell_state.h5ad"

OUTPUT_DIR="${SCRIPT_DIR}/niche_pairs_output"
STATE_COL="cell_states"
CELL_ID_COL="index"  # Column in cell_adata.obs that contains cell IDs matching niche cell_ids

# Morphology extraction flag (set to 0 to disable)
EXTRACT_MORPHOLOGY=1

echo "=============================================="
echo "SPATIA Niche-Level Pairing Pipeline"
echo "=============================================="
echo "Timestamp: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "Morphology extraction: ${EXTRACT_MORPHOLOGY}"
echo "=============================================="

# Activate environment
if command -v conda &> /dev/null; then
    source ~/.bashrc 2>/dev/null || true
    conda activate spatia 2>/dev/null || echo "Using current environment"
fi

# Check Python
which python
python --version

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check for morphology extraction dependencies
if [ "${EXTRACT_MORPHOLOGY}" -eq 1 ]; then
    echo ""
    echo "Checking morphology extraction dependencies..."
    python -c "import cv2; import lmdb; from skimage import measure, filters; from PIL import Image; print('All dependencies available')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Warning: Morphology dependencies not available, disabling morphology extraction"
        EXTRACT_MORPHOLOGY=0
    fi
fi

# Run niche pairing
echo ""
echo "Running grid-based niche pairing..."
echo "=============================================="

cd "${SCRIPT_DIR}"

# Build command with optional LMDB path
CMD="python generate_grid_niche_pairs.py \
    --niche_adata ${NICHE_ADATA_PATH} \
    --cell_adata ${CELL_ADATA_PATH} \
    --out_dir ${OUTPUT_DIR} \
    --state_col ${STATE_COL} \
    --cell_id_col ${CELL_ID_COL} \
    --grid_size 256 \
    --min_cells 5 \
    --min_fraction 0.05 \
    --max_pairs 200 \
    --seed 42"

# Add LMDB path for morphology extraction if enabled
if [ "${EXTRACT_MORPHOLOGY}" -eq 1 ] && [ -f "${NICHE_LMDB_PATH}" ]; then
    CMD="${CMD} --lmdb_path ${NICHE_LMDB_PATH} --dataset_prefix ${DATASET_PREFIX} --include_ot_confidence"
    echo "Morphology extraction: ENABLED"
    echo "OT confidence output: ENABLED"
else
    echo "Morphology extraction: DISABLED"
    echo "OT confidence output: DISABLED (use --include_ot_confidence to enable)"
fi

# Run the command
eval ${CMD}

echo ""
echo "=============================================="
echo "Niche pairing completed!"
echo "=============================================="
echo ""
echo "Output files:"
ls -la "${OUTPUT_DIR}/"
echo ""
echo "Generated files:"
echo "  - niche_pairs.csv         : Paired grid IDs with metadata and ot_confidence"
echo "  - niche_delta_g.npz       : Per-transition Δg vectors"
if [ "${EXTRACT_MORPHOLOGY}" -eq 1 ]; then
    echo "  - niche_delta_m.npz       : Per-transition Δm vectors (morphology)"
fi
echo "  - niche_config.json       : Configuration and statistics"
echo ""
echo "LMDB path for images: ${NICHE_LMDB_PATH}"
echo ""
echo "Usage example:"
echo "  from spatia_niche_dataset import SpatiaNicheDataset"
echo "  dataset = SpatiaNicheDataset("
echo "      adata_path='${NICHE_ADATA_PATH}',"
echo "      pairs_csv='${OUTPUT_DIR}/niche_pairs.csv',"
echo "      delta_g_npz='${OUTPUT_DIR}/niche_delta_g.npz',"
if [ "${EXTRACT_MORPHOLOGY}" -eq 1 ]; then
    echo "      delta_m_npz='${OUTPUT_DIR}/niche_delta_m.npz',"
fi
echo "      lmdb_path='${NICHE_LMDB_PATH}',"
echo "  )"
