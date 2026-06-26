#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import torch

sys.path.append(str(Path(__file__).parent.parent))

from training.spatia_embedder import SpatiaEmbedder
from training.xenium_dataloader import setup_xenium_metadata


def setup_xenium_dataset_example():
    print("Setting up Xenium dataset example...")
    
    spatia_model_path = "/path/to/spatia/checkpoints/"
    spatia_vocab_path = "/path/to/spatia/vocab.json"
    spatia_gene_stats = "/path/to/spatia/gene_stats.csv"
    spatial_data_path = "/path/to/xenium/data.h5ad"
    image_dir = "/path/to/xenium/images/"
    output_dir = "/path/to/output/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing SPATIA embedder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    spatia_embedder = SpatiaEmbedder(
        model_path=spatia_model_path,
        vocab_path=spatia_vocab_path,
        gene_stats_path=spatia_gene_stats,
        device=device,
        batch_size=32,
    )
    
    print("Loading spatial data...")
    adata = sc.read_h5ad(spatial_data_path)
    adata = spatia_embedder.preprocess_adata(adata)
    
    print(f"Loaded {adata.n_obs} cells with {adata.n_vars} genes")
    
    print("Generating SPATIA embeddings...")
    cell_embeddings = spatia_embedder.generate_embeddings(adata)
    
    print(f"Generated embeddings shape: {cell_embeddings.shape}")
    
    print("Creating metadata file...")
    metadata_path = os.path.join(output_dir, "xenium_metadata.csv")
    
    metadata_rows = []
    for i, cell_id in enumerate(adata.obs_names[:100]):
        metadata_rows.append({
            'cell_id': cell_id,
            'ctrl_image_filename': f"{cell_id}_ctrl.png",
            'trt_image_filename': f"{cell_id}_trt.png",
            'treatment_id': np.random.randint(0, 10),
            'x_coord': adata.obs.iloc[i].get('x', np.random.rand()),
            'y_coord': adata.obs.iloc[i].get('y', np.random.rand()),
        })
    
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Created metadata file: {metadata_path}")
    
    embeddings_path = os.path.join(output_dir, "spatia_embeddings.npy")
    np.save(embeddings_path, cell_embeddings)
    print(f"Saved embeddings to: {embeddings_path}")
    
    return {
        'metadata_path': metadata_path,
        'embeddings_path': embeddings_path,
        'spatial_data_path': spatial_data_path,
        'image_dir': image_dir,
        'embedding_dim': cell_embeddings.shape[1]
    }


def create_training_config(dataset_info, output_dir):
    config = f"""# Xenium SPATIA Training Configuration
task_name: xenium_spatia_example

# Dataset configuration
dataset_name: 'xenium'
ood_set: []
mol_list: null
trainable_emb: False
n_channels: 3 
multimodal: False
batch_correction: False
batch_key: null
use_condition_embeddings: True
add_controls: False
condition_embedding_dimension: {dataset_info['embedding_dim']}
modality_list: 
  - Spatial

# SPATIA-specific paths
spatia_model_path: "/path/to/spatia/checkpoints/"
spatia_vocab_path: "/path/to/spatia/vocab.json"
spatia_gene_stats: "/path/to/spatia/gene_stats.csv"
spatial_data_path: "{dataset_info['spatial_data_path']}"

# Data paths
image_path: "{dataset_info['image_dir']}"
data_index_path: "{dataset_info['metadata_path']}"
embedding_path: null  # Generated from SPATIA

# Data handling
augment_train: True 
normalize: True
"""
    
    config_path = os.path.join(output_dir, "xenium_spatia_config.yaml")
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"Created training config: {config_path}")
    return config_path


def create_training_script(config_path, output_dir):
    script_content = f"""#!/bin/bash
# Example training script for Xenium SPATIA

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training parameters
BATCH_SIZE=16
EPOCHS=100
LR=1e-4
OUTPUT_DIR="{output_dir}/training_output"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python train_xenium_spatia.py \\
    --config xenium_spatia_config \\
    --batch-size $BATCH_SIZE \\
    --epochs $EPOCHS \\
    --lr $LR \\
    --output-dir $OUTPUT_DIR \\
    --spatia-model-path "/path/to/spatia/checkpoints/" \\
    --spatia-vocab-path "/path/to/spatia/vocab.json" \\
    --spatia-gene-stats "/path/to/spatia/gene_stats.csv" \\
    --spatial-data-path "{dataset_info['spatial_data_path']}" \\
    --image-path "{dataset_info['image_dir']}" \\
    --data-index-path "{dataset_info['metadata_path']}" \\
    --use-initial 1 \\
    --eval-frequency 10 \\
    --save-frequency 10 \\
    --fid-samples 1000 \\
    --device cuda \\
    --distributed
"""
    
    script_path = os.path.join(output_dir, "train_xenium_spatia.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"Created training script: {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Xenium SPATIA Example Setup")
    parser.add_argument('--output-dir', type=str, default='./xenium_spatia_example',
                        help='Output directory for example files')
    args = parser.parse_args()
    
    print("=== Xenium SPATIA Integration Example ===")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        dataset_info = setup_xenium_dataset_example()
        
        config_path = create_training_config(dataset_info, args.output_dir)
        
        script_path = create_training_script(config_path, args.output_dir)
        
        print("\\n=== Setup Complete ===")
        print(f"Dataset info: {dataset_info}")
        print(f"Config file: {config_path}")
        print(f"Training script: {script_path}")
        
        print("\\n=== Next Steps ===")
        print("1. Update the paths in the configuration files to match your data")
        print("2. Ensure your image files are properly organized")
        print("3. Run the training script:")
        print(f"   bash {script_path}")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        print("\\nNote: This is an example script. You need to:")
        print("1. Update all paths to point to your actual data")
        print("2. Ensure SPATIA model checkpoints are available")
        print("3. Prepare your Xenium image and expression data")


if __name__ == "__main__":
    main()
