import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scgpt_spatial.inference import main
from scgpt_spatial.configs import InferenceConfig


def log_info(msg):
    print(f"[INFO] {msg}")


def log_warning(msg):
    print(f"[WARNING] {msg}")


def log_error(msg):
    print(f"[ERROR] {msg}")


def extract_embedding(h5ad_file, model_checkpoint_dir, step, output_path, seed=0):
    config_path = Path(model_checkpoint_dir) / "config.json"
    
    weight_path = Path(model_checkpoint_dir) / f"checkpoint-step-{step}" / "model.safetensors"
    if not weight_path.exists():
        weight_path = Path(model_checkpoint_dir) / "best_model.pt"
        if not weight_path.exists():
            weight_path = Path(model_checkpoint_dir) / "best_model_full.pt"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"Weights not found: {weight_path}")
    if not Path(h5ad_file).exists():
        raise FileNotFoundError(f"Data not found: {h5ad_file}")
    
    log_info(f"Data: {h5ad_file}")
    log_info(f"Config: {config_path}")
    log_info(f"Weights: {weight_path}")
    log_info(f"Output: {output_path}")
    
    infer_config = InferenceConfig(
        seed=seed,
        spatial_config_path=str(config_path),
        spatial_weight_path=str(weight_path),
        h5ad_file=h5ad_file,
        output_path=output_path,
    )
    
    main(infer_config)
    log_info(f"Saved: {output_path}")


def batch_extract_embeddings(datasets, data_dir, model_checkpoint_dir, step, output_dir, seeds=[0, 1, 2, 3]):
    for dataset in datasets:
        log_info(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")
        h5ad_file = f"{data_dir}/{dataset}.h5ad"
        
        for seed in seeds:
            log_info(f"\nSeed {seed}")
            output_path = f"{output_dir}/seed_{seed}/step_{step}/{dataset}.npy"
            
            if Path(output_path).exists():
                log_warning(f"Skip (exists): {output_path}")
                continue
            
            try:
                extract_embedding(h5ad_file, model_checkpoint_dir, step, output_path, seed)
            except Exception as e:
                log_error(f"Failed: {dataset}, seed={seed}\nError: {str(e)}")
                continue


if __name__ == "__main__":
    MODEL_CHECKPOINT_DIR = os.environ.get("SPATIA_CHECKPOINT_DIR", "./checkpoints")
    DATA_DIR = os.environ.get("SPATIA_DATA_DIR", "./data")
    OUTPUT_DIR = os.environ.get("SPATIA_OUTPUT_DIR", "./embeddings")
    
    DATASETS = [
        "HCC_filtered",
    ]
    
    STEPS = [10000, 20000, 30000, 40000, 50000, 60000]
    SEEDS = [0, 1, 2, 3]
    
    for step in STEPS:
        log_info(f"\n{'#'*60}\nCheckpoint step: {step}\n{'#'*60}")
        batch_extract_embeddings(DATASETS, DATA_DIR, MODEL_CHECKPOINT_DIR, step, OUTPUT_DIR, SEEDS)
    
    log_info(f"\n{'#'*60}\nAll extractions complete!\n{'#'*60}")
