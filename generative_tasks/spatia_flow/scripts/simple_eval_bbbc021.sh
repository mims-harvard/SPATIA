#!/bin/bash
# Simple single-GPU execution script (no distributed training)
# ============================================
# USER: Set these paths before running
# ============================================
# OUTPUT_DIR - Directory for outputs and logs
# CKPT       - Path to CellFlux bbbc021 checkpoint file
# ============================================
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_bbbc}"
CKPT="${CKPT:-/path/to/checkpoints/cellflux/bbbc021/checkpoint.pth}"

# Set which GPU to use
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT_DIR"

python train.py \
    --dataset=bbbc021 \
    --config=bbbc021_all \
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
    --resume="$CKPT" \
    --start_epoch=0 \
    --world_size=1 \
    --dist_url="tcp://localhost:12355" >> "$OUTPUT_DIR/log.txt" 2>&1
