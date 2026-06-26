import sys
import argparse
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


def log_success(msg):
    print(f"[SUCCESS] {msg}")


def extract_single(h5ad_file, model_dir, step, output_path, seed=0):
    config_path = Path(model_dir) / "config.json"
    
    weight_path = Path(model_dir) / f"checkpoint-step-{step}" / "model.safetensors"
    if not weight_path.exists():
        weight_path = Path(model_dir) / "best_model.pt"
        if not weight_path.exists():
            weight_path = Path(model_dir) / "best_model_full.pt"
    
    for path, name in [(config_path, "Config"), (weight_path, "Weights"), (h5ad_file, "Data")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    log_info(f"Data: {h5ad_file}")
    log_info(f"Model: {model_dir}, Step: {step}, Seed: {seed}")
    log_info(f"Output: {output_path}")
    
    config = InferenceConfig(
        seed=seed,
        spatial_config_path=str(config_path),
        spatial_weight_path=str(weight_path),
        h5ad_file=str(h5ad_file),
        output_path=str(output_path),
    )
    
    main(config)
    log_success(f"Done: {output_path}")


def extract_batch(h5ad_file, model_dir, steps, seeds, output_dir):
    output_dir = Path(output_dir)
    h5ad_name = Path(h5ad_file).stem
    
    total = len(steps) * len(seeds)
    current = 0
    
    for step in steps:
        for seed in seeds:
            current += 1
            log_info(f"\nProgress: {current}/{total}")
            
            output_path = output_dir / f"seed_{seed}" / f"step_{step}" / f"{h5ad_name}.npy"
            
            if output_path.exists():
                log_warning(f"Skip (exists): {output_path}")
                continue
            
            try:
                extract_single(h5ad_file, model_dir, step, output_path, seed)
            except Exception as e:
                log_error(f"Failed - Step={step}, Seed={seed}: {e}")
                continue
    
    log_success(f"\nBatch extraction complete: {total} embeddings")


def main_cli():
    parser = argparse.ArgumentParser(description="Extract SPATIA embeddings")
    
    parser.add_argument("--h5ad", required=True, help="Input h5ad file")
    parser.add_argument("--model", required=True, help="Model checkpoint directory")
    parser.add_argument("--step", type=int, help="Checkpoint step (single mode)")
    parser.add_argument("--steps", type=int, nargs='+', help="Checkpoint steps (batch mode)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (single mode, default=0)")
    parser.add_argument("--seeds", type=int, nargs='+', help="Random seeds (batch mode)")
    parser.add_argument("--output", required=True, help="Output path (.npy file or directory)")
    
    args = parser.parse_args()
    
    if args.steps and args.seeds:
        log_info("Batch mode: processing multiple steps and seeds")
        extract_batch(args.h5ad, args.model, args.steps, args.seeds, args.output)
    elif args.step is not None:
        log_info("Single mode")
        extract_single(args.h5ad, args.model, args.step, args.output, args.seed)
    else:
        parser.error("Specify --step (single) or --steps and --seeds (batch)")


if __name__ == "__main__":
    main_cli()
