import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors
import torch
import tyro
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scgpt_spatial.configs import MainConfig
from scgpt_spatial.data_collator import DataCollator
from scgpt_spatial.dataset import SingleAdataDataset, create_dataloaders
from scgpt_spatial.model import TransformerModel
from scgpt_spatial.tasks.cell_emb import load_pretrained
from scgpt_spatial.tokenizer import GeneVocab


@dataclass
class Config:
    spatial_config_path: str
    spatial_weight_path: str
    h5ad_file: str
    output_path: str
    seed: int = 0
    spatial_datadir: Optional[str] = None
    fix_missing_images: Optional[bool] = None
    batch_size: int = 128
    num_workers: int = 16
    l2_normalize: bool = True
    gene_col: Optional[str] = None
    platform: Optional[str] = None
    table2_emb_dir: Optional[str] = None


def _load_state_dict(path: str):
    if path.endswith(".safetensors"):
        out = {}
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    return torch.load(Path(path), map_location="cpu")


def _save(arr_list, path: Path, normalize: bool):
    arr = np.concatenate(arr_list, axis=0)
    if normalize:
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    np.save(path, arr)
    logger.info(f"saved {path}  {arr.shape}")
    return arr


def main(cfg: Config):
    set_seed(cfg.seed)
    accelerator = Accelerator()

    mcfg = MainConfig.from_json(Path(cfg.spatial_config_path).read_text())
    spatial_datadir = cfg.spatial_datadir or mcfg.data.spatial_datadir
    fix_missing = (
        cfg.fix_missing_images
        if cfg.fix_missing_images is not None
        else mcfg.data.fix_missing_images
    )
    if spatial_datadir is None:
        raise ValueError(
            "No image LMDB. Table 2 needs the per-cell image crops; pass "
            "--spatial-datadir pointing at the crop LMDB for this dataset."
        )

    vocab = GeneVocab.from_file(Path(mcfg.data.vocab_path))
    for token in [mcfg.trainer.pad_token, "<cls>", "<eoc>"]:
        if token not in vocab:
            vocab.append_token(token)

    gene_col = cfg.gene_col or mcfg.data.gene_col
    logger.info(f"loading dataset (gene_col={gene_col})...")
    dataset = SingleAdataDataset(
        adata_path=cfg.h5ad_file,
        vocab=vocab,
        gene_stats_file=Path(mcfg.data.gene_stats_file),
        gene_col=gene_col,
        cls_token="<cls>",
        pad_value=0.0,
        require_log1p=True,
        spatial_datadir=spatial_datadir,
        fix_missing_images=fix_missing,
        preprocessor_cls=mcfg.data.preprocessor_cls,
        inference=True,
    )

    mask_value = mcfg.data.n_bins + 1 if mcfg.data.input_emb_style == "category" else -1
    pad_value = mcfg.data.n_bins if mcfg.data.input_emb_style == "category" else -2
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[mcfg.trainer.pad_token],
        pad_value=pad_value,
        do_mlm=False,
        do_binning=(mcfg.data.input_style == "binned"),
        n_bins=mcfg.data.n_bins,
        mlm_probability=mcfg.trainer.mask_ratio,
        mask_value=mask_value,
        max_length=mcfg.data.max_seq_len,
        sampling=mcfg.data.trunc_by_sample,
        data_style="pcpt",
    )
    dataloader, _ = create_dataloaders(
        dataset,
        data_collator=collator,
        batch_size=cfg.batch_size,
        shuffle=False,
        validation_split=0,
        num_workers=cfg.num_workers,
    )
    logger.info(f"{len(dataloader)} batches")

    n_input_bins = (
        (mcfg.data.n_bins + 2) if mcfg.data.input_emb_style == "category" else mcfg.data.n_bins
    )
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=mcfg.model.embsize,
        nhead=mcfg.model.nheads,
        d_hid=mcfg.model.d_hid,
        nlayers=mcfg.model.nlayers,
        nlayers_cls=mcfg.model.n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=mcfg.model.dropout,
        pad_token=mcfg.trainer.pad_token,
        pad_value=pad_value,
        do_mvc=True,
        do_dab=False,
        use_generative_training=False,
        use_batch_labels=False,
        explicit_zero_prob=False,
        use_fast_transformer=mcfg.model.use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
        use_MVC_impute=True,
        use_moe_dec=True,
        n_input_bins=n_input_bins,
        input_emb_style=mcfg.data.input_emb_style,
        image_encoder_cls=mcfg.model.image_encoder_cls,
        combine_weight=mcfg.model.combine_weight,
        image_combine_weight=mcfg.model.image_combine_weight,
        image_recon_loss_weight=mcfg.model.image_recon_loss_weight,
    )
    if model.image_encoder_cls is None:
        raise ValueError(
            "Checkpoint has no image encoder; this is the gene-only model. "
            "Table 2 needs the multimodal checkpoint."
        )

    load_pretrained(model, _load_state_dict(cfg.spatial_weight_path), verbose=True)
    model, dataloader = accelerator.prepare(model, dataloader)

    fused, gene, image = [], [], []
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=mcfg.trainer.fp16):
        for batch in tqdm(dataloader, desc="extracting"):
            base = model.module if hasattr(model, "module") else model
            pad_mask = batch["gene"].eq(vocab[mcfg.trainer.pad_token])
            transformer_output = base._encode(
                src=batch["gene"], values=batch["expr"], src_key_padding_mask=pad_mask
            )
            outputs = base._process_transformer_output(
                transformer_output, image=batch["image"]
            )
            fused.append(outputs["expression_latent"][:, 0, :].float().cpu().numpy())
            gene.append(transformer_output[:, 0, :].float().cpu().numpy())
            if outputs["image_latent"] is not None:
                image.append(outputs["image_latent"][:, 0, :].float().cpu().numpy())

    out = Path(cfg.output_path)
    out.mkdir(parents=True, exist_ok=True)
    fused_arr = _save(fused, out / "fused_embeddings.npy", cfg.l2_normalize)
    _save(gene, out / "gene_embeddings.npy", cfg.l2_normalize)
    if image:
        _save(image, out / "image_embeddings.npy", cfg.l2_normalize)
    else:
        logger.warning("no image latents produced (no images?) - fused == gene")

    if cfg.platform and cfg.table2_emb_dir:
        dst = Path(cfg.table2_emb_dir) / "spatia"
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / f"{cfg.platform}_embeddings.npy", fused_arr)
        logger.info(f"table2 layout: {dst / (cfg.platform + '_embeddings.npy')}")


if __name__ == "__main__":
    main(tyro.cli(Config))
