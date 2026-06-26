#!/bin/bash
# ============================================
# USER: Set these paths before running
# ============================================
# OUTPUT_DIR  - Directory containing CellFlux outputs
# IMAGE_ROOT  - Directory containing generated FID samples
# ============================================
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
IMAGE_ROOT="${IMAGE_ROOT:-${OUTPUT_DIR}/fid_samples/epoch-100}"

python eval_fid.py --model_name cellflux --dataset bbbc021 --num_to_cal 5120 \
        --image_root "$IMAGE_ROOT" >> "${OUTPUT_DIR}/eval_bbbc/fid_log.txt" 2>&1

# python eval_fid.py --model_name cellflux --dataset rxrx1 \
#         --image_root /path/to/your/rxrx1/generated_images

# python eval_fid.py --model_name cellflux --dataset cpg0000 \
#         --image_root /path/to/your/cpg0000/generated_images