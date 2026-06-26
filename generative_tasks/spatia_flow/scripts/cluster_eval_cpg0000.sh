#!/bin/bash
# ============================================
# USER: Modify REPO, OUTPUT_DIR, CKPT for your cluster
# ============================================
# REPO       - Path to the CellFlux directory
# OUTPUT_DIR - Directory for outputs
# CKPT       - Path to CellFlux cpg0000 checkpoint file
# ============================================

REPO="${REPO:-/path/to/SPATIA/generative_tasks/spatia_flow}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
CKPT="${CKPT:-/path/to/checkpoints/cellflux/cpg0000/checkpoint.pth}"

# Create log directory if it doesn't exist
mkdir -p logs/eval_cpg0000

# Adjust module loading for your cluster
# module load python/3.9
# module load cuda/11.8

# Set GPU visibility - using both GPUs since your cluster has --gres=gpu:2
export CUDA_VISIBLE_DEVICES=0,1

# Change to the correct directory
cd "$REPO"

# Run with torchrun for proper multi-GPU distributed training
torchrun --nproc_per_node=2 --master_port=12356 train.py \
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
