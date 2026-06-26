#!/bin/bash
# ============================================
# REPO       - Path to the SPATIA-scgpt directory
# DATA_DIR   - Path to your datasets directory
# ============================================

eval "$(conda shell.bash hook)"
conda activate spatia

set -euo pipefail
set -x

REPO="${REPO:-/path/to/SPATIA/gene_encoders/SPATIA-scgpt}"
DATA_DIR="${DATA_DIR:-/path/to/data}"

export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export ARRAY_INDEX=$SLURM_ARRAY_TASK_ID

combine_weights=(0 1 1e-8 1e-6 1e-4 1e-2)
export COMBINE_WEIGHT=${combine_weights[$ARRAY_INDEX]}
export IMAGE_COMBINE_WEIGHT=${combine_weights[$ARRAY_INDEX]}
export MASTER_PORT=$((29500 + RANDOM % 1000))

export DATE=$(date +%m%d)
export DATASET_NAME="COAD"

export DATA_SOURCE="./data/${DATASET_NAME}"
export SAVE_DIR="./save/${DATE}_${DATASET_NAME}"
export TEST_H5AD_PATH="${DATA_DIR}/COAD_filtered_10k.h5ad"
export WANDB_NAME="testcls_${DATASET_NAME}_expr${COMBINE_WEIGHT}_image${IMAGE_COMBINE_WEIGHT}_auxloss1.0"

cd "$REPO"
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 --main_process_port $MASTER_PORT \
    -m scgpt_spatial.train \
    --data.data-source $DATA_SOURCE \
    --data.vocab-path ./checkpoints/vocab.json \
    --data.gene-stats-file ./checkpoints/all_dict_mean_std.csv \
    --trainer.save-dir $SAVE_DIR \
    --trainer.batch-size 32 \
    --model.load_model_path ./checkpoints/ \
    --data.spatial_datadir "${DATA_DIR}/lmdb/xenium_multiscale_0514.lmdb" \
    --data.preprocessor_cls "facebook/vit-mae-base" \
    --model.image_encoder_cls "facebook/vit-mae-base" \
    --model.combine_weight $COMBINE_WEIGHT \
    --model.image_combine_weight $IMAGE_COMBINE_WEIGHT \
    --model.image_recon_loss_weight 1 \
    --data.test-h5ad-path $TEST_H5AD_PATH
