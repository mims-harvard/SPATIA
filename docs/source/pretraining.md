# Representation Training (Stage 1)

SPATIA offers two gene encoder backends. **SPATIA-scprint** (primary, scPRINT-based) uses a Flash Transformer with ViT-MAE image cross-attention. **SPATIA-scgpt** (alternative) uses a scGPT-style transformer with mixture-of-experts layers.

---

## Prerequisites

Before training, you need:

1. **MIST dataset** — constructed via the [Dataset Construction](dataset.md) pipeline (lamindb Collection + LMDB)
2. **Base scPRINT checkpoint** — download with:
   ```bash
   cd gene_encoders/SPATIA-scprint/data/model && bash download.sh
   ```
3. **Gene embeddings** — `data/generated/gene_embeddings.parquet` (from the scPRINT pretrained model)
4. **Biomart gene positions** — `data/main/biomart_pos.parquet` (shipped in this repo)

---

## SPATIA-scprint (Primary)

### Configure paths

Edit `gene_encoders/SPATIA-scprint/config/0510_base_spatial_all_crossattention_medium.yml`:

```yaml
data:
  spatial_datadir: /your/path/to/lmdb/all.lmdb     # LMDB from Stage C
  collection_name: xenium_all_0212                   # lamindb Collection from Stage B
  gene_embeddings: ./data/generated/gene_embeddings.parquet
  do_gene_pos: ./data/main/biomart_pos.parquet
model:
  ckpt_path: ./data/model/scPRINT/medium.ckpt       # base scPRINT checkpoint
  clip_model_type: "facebook/vit-mae-base"           # ViT-MAE image encoder
```

### Run training

```bash
cd gene_encoders/SPATIA-scprint

scprint_spatial fit \
    --config config/0510_base_spatial_all_crossattention_medium.yml \
    --model.ckpt_path data/model/scPRINT/medium.ckpt \
    --data.batch_size 8
```

Training uses PyTorch Lightning with bf16 mixed precision, DDP strategy, and Weights & Biases logging.

### Extract embeddings (SPATIA-scprint)

```bash
python prediction_tasks/scripts/extract_spatia_embeddings.py \
    --checkpoint /path/to/scprint_checkpoint.ckpt \
    --adata_path /path/to/dataset.h5ad \
    --lmdb_dir /path/to/images.lmdb \          # omit for gene-only mode
    --output_path embeddings/spatia_embeddings.npy
```

---

## SPATIA-scgpt (Alternative)

### Run training

```bash
cd gene_encoders/SPATIA-scgpt
bash scgpt_spatial/0704_train_spatial_4h100.sh
```

### Extract embeddings (SPATIA-scgpt)

```bash
python gene_encoders/SPATIA-scgpt/tutorials/extract_multimodal_embeddings.py \
    --spatial-config-path /path/to/config.json \
    --spatial-weight-path /path/to/best_model.pt \
    --h5ad-file /path/to/dataset.h5ad \
    --spatial-datadir /path/to/images.lmdb \
    --output-path /path/to/output
```

---

## Architecture Summary

```{image} ../../static/images/model_main.png
:width: 600px
:align: center
```

| Component | Detail |
|-----------|--------|
| Gene encoder | Flash Transformer, d_model 256, 8 layers |
| Image encoder | ViT-MAE base (12 enc / 8 dec layers, patch 16) |
| Fusion | Cross-attention (image queries, gene keys/values) |
| Image init | `facebook/vit-mae-base` (ImageNet-1k) |
| Gene init | scPRINT pretrained (mandatory) |
| Total params | ~155M |
| Training precision | bf16-mixed |
