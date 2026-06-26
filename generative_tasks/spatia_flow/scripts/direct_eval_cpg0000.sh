#!/bin/bash
# Direct execution script for CPG0000 dataset evaluation (no SLURM)
# ============================================
# USER: Set these paths before running
# ============================================
# OUTPUT_DIR - Directory for outputs
# CKPT       - Path to CellFlux cpg0000 checkpoint file
# ============================================
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
CKPT="${CKPT:-/path/to/checkpoints/cellflux/cpg0000/checkpoint.pth}"

# Set which GPU to use - modify this to match your available GPUs
export CUDA_VISIBLE_DEVICES=0

# Set up environment variables for single-GPU distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12356
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

python train.py \
    --dataset=cpg0000 \
    --config=cpg0000 \
    --batch_size=32 \
    --accum_iter=1 \
    --eval_frequency=10 \
    --epochs=3000 \
    --class_drop_prob=0.2 \
    --cfg_scale=0.2 \
    --compute_fid \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --fid_samples=30720 \
    --output_dir="$OUTPUT_DIR" \
    --use_initial=2 \
    --eval_only \
    --noise_level=1.0 \
    --save_fid_samples \
    --resume="$CKPT"
