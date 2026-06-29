# MIST Dataset Construction

MIST (Multimodal Imaging and Spatial Transcriptomics) is the pretraining
dataset. It combines cell-level gene expression with morphology image crops
from Xenium spatial transcriptomics data. Construction has three stages:

## Stage A: Crop cell images into LMDB

Crop cell-centered patches from Xenium morphology TIFF images and store them
in an LMDB database.

**Input layout** (per Xenium dataset):
```
dataset_dir/
├── morphology_mip.ome.tif    # or morphology.ome.tif / DAPI.tif
├── cells.parquet             # cell centroids + boundaries
└── cell_feature_matrix.h5    # (optional, for building h5ad)
```

**Crop pipeline:**
```bash
cd gene_encoders/SPATIA-scprint

# Standard Xenium format (cells.parquet + morphology.ome.tif)
python scripts/0510_crop_images_cell_refactored.py \
    --output-lmdb /path/to/output/dataset_name.lmdb \
    --output-size 256 \
    --cache /path/to/cache

# SPATCH format (adata.h5ad + DAPI.tif, for COAD/HCC/OV datasets)
python scripts/0510_crop_images_cell_spatch.py \
    --input-dir /path/to/SPATCH/Xenium-5K \
    --output-lmdb /path/to/output/lmdb \
    --dataset-name HCC
```

**Image processing details:**
- TIFF max intensity projection across channels
- Normalize to uint8 (0-255)
- Crop around cell centroid (adaptive size from cell boundaries, or default 32px radius)
- Resize to 256x256
- Store as raw bytes in LMDB with key format: `{dataset_name}/{cell_id}`
- Coordinate mapping: `pixel_x = spatial_y / 0.2125`, `pixel_y = spatial_x / 0.2125` (Xenium coordinate swap)

## Stage B: Build annotated h5ad and register in lamindb

Annotate each dataset with ontology metadata and add to a lamindb Collection:

```bash
cd gene_encoders/SPATIA-scprint

python scripts/0512_add_single_dataset.py \
    /path/to/adata.h5ad \
    --tissue lung --disease normal \
    --dataset_name xenium_lung \
    --collection_name xenium_all_0212
```

Required metadata columns (added automatically by the script):
- `organism_ontology_term_id` (e.g., `NCBITaxon:9606`)
- `cell_type_ontology_term_id`, `tissue_ontology_term_id`, `disease_ontology_term_id`
- `assay_ontology_term_id`, `sex_ontology_term_id`, `development_stage_ontology_term_id`
- `donor_id`, `dataset_name`, `index` (cell ID matching LMDB keys)

## Stage C: Merge per-dataset LMDBs (optional)

Consolidate multiple per-dataset LMDBs into a single file:

```bash
python scripts/0514_merge_lmdb.py \
    --input-dir /path/to/per_dataset_lmdbs/ \
    --output /path/to/merged/all.lmdb
```

## Data loading at training time

The training data loader (`scdataloader.data_spatial.Dataset`) reads:
1. Gene expression from a lamindb Collection (multiple h5ad files)
2. Cell images from LMDB files (supports multiple scales)

LMDB environments are mapped to image keys in the batch:
- 1st LMDB path -> `image` (cell-level crop)
- 2nd LMDB path -> `region_image` (niche-level, optional)
- 3rd LMDB path -> `tissue_image` (tissue-level, optional)

Images are preprocessed at load time: 256x256 grayscale -> stack to RGB ->
`AutoImageProcessor` (ViT-MAE) -> `(3, 224, 224)` float tensor.
