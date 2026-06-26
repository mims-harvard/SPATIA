import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import ipdb
import numpy as np
import safetensors
import torch
import tyro
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .configs import InferenceConfig, MainConfig
from .data_collator import DataCollator
from .dataset import SingleAdataDataset, create_dataloaders
from .model import TransformerModel
from .tasks.cell_emb import load_pretrained
from .tokenizer import GeneVocab


def load_safetensors(file_path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    tensors = {}
    with safetensors.safe_open(file_path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def setup_logging_and_saving(cfg: MainConfig) -> Tuple[Path, Optional[SummaryWriter]]:
    if cfg.trainer.exp_name is None:
        cfg.trainer.exp_name = (
            f"scgpt-inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    save_dir = Path(cfg.trainer.save_dir) / cfg.trainer.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w") as f:
        serializable_config = asdict(cfg)
        json.dump(serializable_config, f, indent=4)

    logger.add(save_dir / "run.log")
    logger.info(f"Configuration saved to {save_dir / 'config.json'}")

    return save_dir


def main(infer_cfg: InferenceConfig) -> None:
    set_seed(infer_cfg.seed)

    accelerator = Accelerator()

    cfg = MainConfig.from_json(Path(infer_cfg.spatial_config_path).read_text())

    if cfg.model.load_model_path:
        args_json_path = Path(cfg.model.load_model_path) / "args.json"
        if args_json_path.exists():
            logger.info(f"Loading model args from {args_json_path}")
            with open(args_json_path, "r") as f:
                loaded_args = json.load(f)

            updated_params = []
            for key, value in loaded_args.items():
                target_key = key
                if key == "fast_transformer":
                    target_key = "use_fast_transformer"
                if hasattr(cfg.model, target_key):
                    current_value = getattr(cfg.model, target_key)
                    setattr(cfg.model, target_key, value)
                    updated_params.append(
                        f"model.{target_key} (from args.json key '{key}') {current_value} -> {value}"
                    )

            if updated_params and accelerator.is_main_process:
                logger.info(f"Updated model config from {args_json_path}:")
                for param_update in updated_params:
                    logger.info(f"  - {param_update}")
        else:
            if accelerator.is_main_process:
                logger.warning(
                    f"Warning: args.json not found at {args_json_path}. Using model config from command line or defaults."
                )

    if accelerator.is_main_process:
        setup_logging_and_saving(cfg)

    vocab = GeneVocab.from_file(Path(cfg.data.vocab_path))
    special_tokens = [cfg.trainer.pad_token, "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
            logger.info(f"Added special token '{s}' to vocabulary.")

    logger.info("Loading data...")

    full_dataset = SingleAdataDataset(
        adata_path=infer_cfg.h5ad_file,
        vocab=vocab,
        gene_stats_file=Path(cfg.data.gene_stats_file),
        gene_col=cfg.data.gene_col,
        cls_token="<cls>",
        pad_value=0.0,
        require_log1p=True,
        spatial_datadir=cfg.data.spatial_datadir,
        fix_missing_images=cfg.data.fix_missing_images,
        preprocessor_cls=cfg.data.preprocessor_cls,
        inference=True,
    )

    if cfg.data.input_emb_style == "category":
        mask_value = cfg.data.n_bins + 1
        pad_value = cfg.data.n_bins
    else:
        mask_value = -1
        pad_value = -2

    if cfg.trainer.training_tasks in ["gen", "both"]:
        if isinstance(cfg.trainer.mask_ratio, float):
            mask_ratio = [0.25, 0.50, 0.75]
            logger.warning(
                "Using multiple mask ratios for generative training: "
                f"{mask_ratio} instead of <default_mask_ratio>."
            )
        else:
            mask_ratio = cfg.trainer.mask_ratio
    else:
        mask_ratio = cfg.trainer.mask_ratio

    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[cfg.trainer.pad_token],
        pad_value=pad_value,
        do_mlm=False,
        do_binning=(cfg.data.input_style == "binned"),
        n_bins=cfg.data.n_bins,
        mlm_probability=mask_ratio,
        mask_value=mask_value,
        max_length=cfg.data.max_seq_len,
        sampling=cfg.data.trunc_by_sample,
        data_style="pcpt",
    )

    train_loader, _ = create_dataloaders(
        full_dataset,
        data_collator=collator,
        batch_size=128,
        shuffle=False,
        validation_split=0,
        num_workers=16,
    )
    logger.info(f"Data loading complete. Train batches: {len(train_loader)}")

    n_input_bins = (
        (cfg.data.n_bins + 2)
        if cfg.data.input_emb_style == "category"
        else cfg.data.n_bins
    )

    num_types = 1
    if not cfg.model.no_cls:
        logger.info(
            "Warning: Automatic cell type counting is not fully supported with the new dataset. "
            "Using a placeholder value. Please verify your CLS head."
        )

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg.model.embsize,
        nhead=cfg.model.nheads,
        d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers,
        nlayers_cls=cfg.model.n_layers_cls,
        n_cls=num_types,
        vocab=vocab,
        dropout=cfg.model.dropout,
        pad_token=cfg.trainer.pad_token,
        pad_value=pad_value,
        do_mvc=True,
        do_dab=False,
        use_generative_training=False,
        use_batch_labels=False,
        explicit_zero_prob=False,
        use_fast_transformer=cfg.model.use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
        use_MVC_impute=True,
        use_moe_dec=True,
        n_input_bins=n_input_bins,
        input_emb_style=cfg.data.input_emb_style,
        image_encoder_cls=cfg.model.image_encoder_cls,
        combine_weight=cfg.model.combine_weight,
        image_combine_weight=cfg.model.image_combine_weight,
        image_recon_loss_weight=cfg.model.image_recon_loss_weight,
    )

    if infer_cfg.spatial_weight_path.endswith(".pt"):
        state_dict = torch.load(Path(infer_cfg.spatial_weight_path), map_location="cpu")
    elif infer_cfg.spatial_weight_path.endswith(".safetensors"):
        state_dict = load_safetensors(
            Path(infer_cfg.spatial_weight_path),
            device="cpu",
        )
    else:
        raise ValueError(
            "Unsupported weight file format. Please provide a .pt or .safetensors file."
        )
    logger.info(f"Loading model from {infer_cfg.spatial_weight_path}")
    load_pretrained(model, state_dict, verbose=True)

    model, train_loader = accelerator.prepare(model, train_loader)

    logger.info("Generating cell embeddings...")
    all_embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.trainer.fp16):
        for batch in tqdm(train_loader, desc="Embedding cells"):
            input_dict = {
                "src": batch["gene"],
                "values": batch["expr"],
                "src_key_padding_mask": batch["gene"].eq(vocab[cfg.trainer.pad_token]),
                "image": batch["image"],
            }

            outputs = model._encode_spatial(
                **input_dict
            )

            embeddings = outputs["expression_latent"]
            embeddings = embeddings[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    logger.info(f"Successfully generated embeddings of shape: {embeddings.shape}")
    save_folder = Path(infer_cfg.output_path).parent
    save_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving embeddings to {infer_cfg.output_path}")
    np.save(
        infer_cfg.output_path,
        embeddings,
    )


if __name__ == "__main__":
    cfg = tyro.cli(InferenceConfig)
    main(cfg)
