# ============================================
# USER: Set these paths before running
# ============================================
# REPO       - Path to the CellFlux directory
# OUTPUT_DIR - Directory for job outputs
# CKPT       - Path to CellFlux cpg0000 checkpoint file
# ============================================
REPO="${REPO:-/path/to/SPATIA/generative_tasks/spatia_flow}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/outputs}"
CKPT="${CKPT:-/path/to/checkpoints/cellflux/cpg0000/checkpoint.pth}"

python submitit_train.py \
    --partition=your_partition \
    --account=your_account \
    --constraint "" \
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
    --job_dir="$OUTPUT_DIR" \
    --shared_dir="$REPO" \
    --use_initial=2 \
    --eval_only \
    --noise_level=1.0 \
    --save_fid_samples \
    --resume="$CKPT" \
    --ngpus=2 \