#!/bin/bash
# ============================================
# REPO - Path to the SPATIA-scprint directory
# ============================================

# export CUDA_VISIBLE_DEVICES=0
set -ex

REPO="${REPO:-/path/to/SPATIA/gene_encoders/SPATIA-scprint}"
cd "$REPO"

combine_weights=("1.0" "0.0" "1e-8" "1e-4" "1e-7" "1e-6" "1e-5" "1e-3" "1e-2" "1e-1")
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=2
export COMBINE_WEIGHT=${combine_weights[$SLURM_ARRAY_TASK_ID]}
export IMAGE_COMBINE_WEIGHT=${COMBINE_WEIGHT}
export IMAGE_RECON_LOSS_WEIGHT=${IMAGE_RECON_LOSS_WEIGHT:-"0.1"}

export DATE=0510
export NODE_TYPE="A100"
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export WANDB_NAME=${DATE}_weight${COMBINE_WEIGHT}_base_spatial_multiscale

scprint_spatial fit --config config/0510_base_spatial_multiscale_medium.yml --scprint_training.name ${DATE}_base_spatial_multiscale_weight${COMBINE_WEIGHT}_${NODE_TYPE} --model.combine_weight ${COMBINE_WEIGHT} --model.image_combine_weight=${IMAGE_COMBINE_WEIGHT} --model.image_recon_loss_weight=${IMAGE_RECON_LOSS_WEIGHT} --data.batch_size 112 --model.ckpt_path data/model/scPRINT/medium.ckpt --data.collection_name xenium_multiscale_v7_1024
