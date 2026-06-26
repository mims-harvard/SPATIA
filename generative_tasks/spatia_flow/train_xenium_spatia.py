#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"

import datetime
import gc
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from tqdm import tqdm

from models.model_configs import MODEL_CONFIGS, instantiate_model
from train_arg_parser import get_args_parser
from training import distributed_mode
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import my_train_one_epoch
from training.spatia_bio_dataloader import create_spatia_bio_dataloader

logger = logging.getLogger(__name__)


def get_spatia_args_parser():
    parser = get_args_parser()
    
    parser.add_argument('--spatia-model-path', type=str,
                        help='Path to SPATIA model checkpoint directory')
    parser.add_argument('--spatia-vocab-path', type=str,
                        help='Path to SPATIA vocabulary JSON file')
    parser.add_argument('--spatia-gene-stats', type=str,
                        help='Path to SPATIA gene statistics CSV file')
    
    parser.add_argument('--adata-path', type=str,
                        help='Path to AnnData h5ad file')
    parser.add_argument('--pairs-csv', type=str,
                        help='Path to paired cells CSV file')
    parser.add_argument('--lmdb-path', type=str,
                        help='Path to LMDB image database')
    parser.add_argument('--delta-g-npz', type=str, default=None,
                        help='Path to delta_g signatures NPZ file (optional)')
    
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Maximum number of pairs for testing (optional)')
    
    return parser


def load_yaml_config(config_name: str) -> dict:
    yaml_path = Path(__file__).parent / "configs" / f"{config_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def merge_args_with_yaml(args, yaml_config: dict):
    args_dict = vars(args)
    
    override_defaults = {
        'dataset', 'class_drop_prob', 'lr', 'batch_size', 'epochs',
        'use_initial', 'noise_level', 'noise_prob', 'skewed_timesteps',
        'edm_schedule', 'ode_method', 'cfg_scale', 'use_ema',
        'lambda_morph', 'morph_proxy_checkpoint',
    }
    
    for key, value in yaml_config.items():
        arg_key = key.replace('-', '_')
        
        if arg_key not in args_dict or args_dict[arg_key] is None or arg_key in override_defaults:
            args_dict[arg_key] = value
    
    return SimpleNamespace(**args_dict)


def validate_args(args):
    required_spatia = ['spatia_model_path', 'spatia_vocab_path', 'spatia_gene_stats']
    required_data = ['adata_path', 'pairs_csv', 'lmdb_path']
    
    missing = []
    for arg in required_spatia + required_data:
        if not hasattr(args, arg) or getattr(args, arg) is None:
            missing.append(arg)
    
    if missing:
        raise ValueError(
            f"Missing required arguments: {missing}\n"
            f"Set them via command line or in the YAML config file."
        )


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    distributed_mode.init_distributed_mode(args)

    logger.info(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info(f"Arguments:\n{args}".replace(", ", ",\n"))
    
    if distributed_mode.is_main_process() and args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f, indent=4, default=str)

    device = torch.device(args.device)

    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger.info("Initializing SPATIA Biological DataLoader...")
    args.num_tasks = distributed_mode.get_world_size()
    args.global_rank = distributed_mode.get_rank()
    
    datamodule = create_spatia_bio_dataloader(args, device)
    data_loader_train = datamodule.train_dataloader()
    data_loader_test = datamodule.test_dataloader()
    
    logger.info(f"Training samples: {len(data_loader_train.dataset)}")
    logger.info(f"Test samples: {len(data_loader_test.dataset)}")
    logger.info(f"SPATIA embedding dimension: {datamodule.latent_dim}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        logger.info(f"GPU memory before model init: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    logger.info("Initializing Model...")
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
    )
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    logger.info(f"Learning rate: {args.lr:.2e}")
    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    if args.use_initial in [1, 2]:
        logger.info("Generating from control image!")
    else:
        logger.info("Generating from random noise!")

    spatia_conditioner = None
    if getattr(args, 'use_spatia_conditioner', False):
        from models.spatia_conditioner import SpatiaConditioner
        conditioner_cfg = getattr(args, 'spatia_conditioner_config', {})
        spatia_conditioner = SpatiaConditioner(
            num_transitions=conditioner_cfg.get('num_transitions', 10),
            dim_g=conditioner_cfg.get('dim_g', 512),
            dim_m=conditioner_cfg.get('dim_m', 10),
            embed_dim=conditioner_cfg.get('embed_dim', 512),
            ctrl_embed_dim=conditioner_cfg.get('ctrl_embed_dim', 512),
            dropout=conditioner_cfg.get('dropout', 0.1),
        ).to(device)
        optimizer.add_param_group({
            'params': spatia_conditioner.parameters(),
            'lr': args.lr,
        })
        logger.info(f"SpatiaConditioner initialized: dim_g={conditioner_cfg.get('dim_g')}, "
                     f"dim_m={conditioner_cfg.get('dim_m')}, "
                     f"num_transitions={conditioner_cfg.get('num_transitions')}")

    feature_extractor = None
    if getattr(args, 'lambda_morph', 0.0) > 0:
        morph_proxy_checkpoint = getattr(args, 'morph_proxy_checkpoint', None)
        if morph_proxy_checkpoint and os.path.exists(morph_proxy_checkpoint):
            from training.train_morph_proxy import MorphProxyEncoder
            ckpt = torch.load(morph_proxy_checkpoint, map_location=device)
            morph_dim = ckpt.get('output_dim', 10)
            feature_extractor = MorphProxyEncoder(output_dim=morph_dim).to(device)
            feature_extractor.load_state_dict(ckpt['model_state_dict'])
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad = False
            logger.info(f"MorphProxyEncoder loaded from {morph_proxy_checkpoint} "
                        f"(dim={morph_dim}, R²={ckpt.get('r2', 'N/A')})")
        else:
            from training.train_loop import FeatureExtractor
            feature_extractor = FeatureExtractor(output_dim=256).to(device)
            feature_extractor.eval()
            logger.info("WARNING: Using random FeatureExtractor (no morph_proxy_checkpoint)")

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
        spatia_conditioner=spatia_conditioner,
    )

    logger.info(f"Starting training from epoch {args.start_epoch} to {args.epochs}")
    start_time = time.time()
    
    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Training"):
        if args.distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
            
        if not args.eval_only:
            train_stats = my_train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                datamodule=datamodule,
                use_initial=args.use_initial,
                spatia_conditioner=spatia_conditioner,
                feature_extractor=feature_extractor,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {"epoch": epoch}

        should_eval = (
            args.output_dir and
            ((args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
             or args.eval_only
             or args.test_run)
        )
        
        if should_eval:
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    spatia_conditioner=spatia_conditioner,
                )
                
            if args.distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
                data_loader_train.sampler.set_epoch(0)
                
            num_tasks = args.num_tasks
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (args.fid_samples // num_tasks)
            else:
                fid_samples = args.fid_samples // num_tasks
            
            eval_stats = eval_model(
                model,
                data_loader_test,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
                datamodule=datamodule,
                use_initial=args.use_initial,
                interpolate=args.interpolate,
                spatia_conditioner=spatia_conditioner,
            )
            
            if eval_stats:
                log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})
                logger.info(log_stats)
                
        if args.output_dir and distributed_mode.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    parser = get_spatia_args_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'config') and args.config:
        yaml_config = load_yaml_config(args.config)
        args = merge_args_with_yaml(args, yaml_config)
    
    if not hasattr(args, 'dataset') or args.dataset not in MODEL_CONFIGS:
        args.dataset = 'xenium'
    
    validate_args(args)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
