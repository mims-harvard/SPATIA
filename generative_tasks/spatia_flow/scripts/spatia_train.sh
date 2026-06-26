# ============================================
# USER: Set REPO to the CellFlux directory
# ============================================
REPO="${REPO:-/path/to/SPATIA/generative_tasks/spatia_flow}"
cd "$REPO"
conda activate spatia

{
    echo "=== SPATIA-CellFlux Training Started at $(date) ==="
    python train_xenium_spatia.py \
        --config spatia_bio \
        --device cuda \
        --batch_size 4 \
        --epochs 100 \
        --output_dir ./outputs/spatia_bio_training/ \
        --eval_frequency 10 \
        --save_frequency 10 \
        --fid_samples 100 \
        --world_size 1 \
        --dist_url "tcp://127.0.0.1:23456" \
        --max_pairs 100
    echo "=== SPATIA-CellFlux Training Completed at $(date) ==="
} > ./outputs/spatia_bio_training/training_log.txt 2>&1
