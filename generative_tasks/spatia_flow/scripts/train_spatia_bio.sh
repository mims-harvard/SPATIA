#!/bin/bash
# ==============================================================================
# SPATIA-CellFlux Training Script
# ==============================================================================
# This script trains a flow matching model for biological perturbation prediction
# using SPATIA embeddings as conditioning.
#
# Usage:
#   bash scripts/train_spatia_bio.sh
#
# For debugging with limited data:
#   bash scripts/train_spatia_bio.sh --max-pairs 50 --test_run
# ==============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default parameters (can be overridden via command line)
CONFIG="spatia_bio"
BATCH_SIZE=1  # Very small batch size for memory efficiency
EPOCHS=100
OUTPUT_DIR="./outputs/spatia_bio_training"
EVAL_FREQUENCY=10
FID_SAMPLES=100
DEVICE="cuda"
NUM_GPUS=2  # Number of GPUs to use
ACCUM_ITER=4  # Gradient accumulation for effective batch size of 8

# Parse command line arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --accum_iter)
            ACCUM_ITER="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "SPATIA-CellFlux Training"
echo "=============================================="
echo "Config: $CONFIG"
echo "Batch size: $BATCH_SIZE (per GPU)"
echo "Accum iter: $ACCUM_ITER"
echo "Effective batch size: $((BATCH_SIZE * ACCUM_ITER * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Num GPUs: $NUM_GPUS"
echo "Extra args: $EXTRA_ARGS"
echo "=============================================="

# Run training with torchrun for distributed training support
torchrun --nproc_per_node=$NUM_GPUS --master_port=12357 train_xenium_spatia.py \
    --config "$CONFIG" \
    --batch_size "$BATCH_SIZE" \
    --accum_iter "$ACCUM_ITER" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --eval_frequency "$EVAL_FREQUENCY" \
    --fid_samples "$FID_SAMPLES" \
    --device "$DEVICE" \
    --use_initial 1 \
    $EXTRA_ARGS

echo "=============================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=============================================="
