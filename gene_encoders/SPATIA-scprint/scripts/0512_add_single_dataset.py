import argparse
import json
import os
from glob import glob

import lamindb as ln
import numpy as np
import scanpy as sc
from filelock import FileLock
from scdataloader import DataModule, Preprocessor
from scdataloader import utils
from scdataloader import utils as data_utils
from tqdm.auto import tqdm
from utils import cell_mapping, disease_mapping, tissue_mapping

os.environ["TMPDIR"] = "/tmp"

parser = argparse.ArgumentParser(description="Annotate and process a single dataset.")
parser.add_argument("adata_path", type=str, help="Path to the anndata file.")
parser.add_argument(
    "--tissue", type=str, default=None, help="Tissue type of the dataset."
)
parser.add_argument(
    "--disease", type=str, default=None, help="Disease type of the dataset."
)
parser.add_argument(
    "--dataset_name", type=str, default=None, help="Name of the dataset."
)
parser.add_argument(
    "--collection_name",
    type=str,
    default=None,
    help="Name of the collection to add the dataset to.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Whether to save the processed dataset to local disk.",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    default=False,
    help="Whether to run the script without making any changes.",
)
args = parser.parse_args()


if args.collection_name is None or args.dry_run:
    print("No collection name provided or dry run enabled, do not check database.")
else:
    with FileLock("lamindb.lock"):
        try:
            collection = ln.Collection.filter(name=args.collection_name).one()

            if collection.artifacts.filter(description=args.adata_path).exists():
                print(
                    f"Dataset from path {args.adata_path} is already in collection {args.collection_name}. Skipping addition."
                )
                exit(0)
        except ln.core.exceptions.DoesNotExist:
            print(
                f"Error: Collection '{args.collection_name}' does not exist. Cannot add dataset."
            )
        except Exception as e:
            print(f"An error occurred during database operations: {e}")

metadata = json.load(open("dataset/metadata.json"))
tissues = []
disease = []
for sample in metadata.values():
    tissues.append(sample["tissue"])
    disease.append(sample["disease"])
print(set(tissues))
print(set(disease))

adata = sc.read_h5ad(args.adata_path)
folder = os.path.dirname(args.adata_path)

if adata.X.max() - int(adata.X.max()) > 0:
    raise ValueError("adata.X should be raw counts, not normalized counts.")
if isinstance(adata.X, np.ndarray):
    print("do not convert since it is already a numpy array")
else:
    print("converting to numpy array")
    adata.X = adata.X.toarray()

if args.dataset_name is None:
    args.dataset_name = folder.split("/")[-1]
if args.tissue is None:
    args.tissue = metadata[args.dataset_name]["tissue"]
if args.tissue == "auto":
    if args.dataset_name in metadata:
        args.tissue = metadata[args.dataset_name]["tissue"]
    else:
        args.tissue = "human"
if args.disease is None:
    args.disease = metadata[args.dataset_name]["disease"]
if args.disease == "auto":
    if args.dataset_name in metadata:
        args.disease = metadata[args.dataset_name]["disease"]
    else:
        args.disease = "unknown"
print(f"Metadata: {args.dataset_name}, {args.tissue}, {args.disease}")
print(f"Example obs names: {adata.obs_names[:5]}")
print(f"Example var names: {adata.var_names[:5]}")
print(f"Obs columns: {adata.obs.columns}")
print(f"Var columns: {adata.var.columns}")

if "organism_ontology_term_id" not in adata.obs.columns:
    adata.var["gene_index"] = adata.var.index
    adata.var.index = adata.var["gene_ids"]

    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata.obs["assay_ontology_term_id"] = "EFO:0022615"
    adata.obs["self_reported_ethnicity_ontology_term_id"] = "unknown"
    adata.obs["sex_ontology_term_id"] = "unknown"
    adata.obs["development_stage_ontology_term_id"] = "unknown"

    adata.obs["donor_id"] = args.dataset_name
    adata.obs["cell_type_ontology_term_id"] = "unknown"

    adata.obs["tissue_ontology_term_id"] = tissue_mapping[args.tissue]
    adata.obs["disease_ontology_term_id"] = disease_mapping[args.disease]
else:
    print(f"Dataset seems to be already annotated, skipping annotation step.")


if any(
    x not in adata.obs.columns
    for x in ["index", "dataset_name", "development_stage_ontology_term_id"]
):
    adata.obs["index"] = adata.obs.index.astype(str)
    adata.obs["dataset_name"] = args.dataset_name
    adata.obs["development_stage_ontology_term_id"] = "HsapDv:0000266"
    print(f"Example index column: {adata.obs['index'][:5]}")
    print(f"Example dataset_name column: {adata.obs['dataset_name'][:5]}")

    preprocessor = Preprocessor(
        do_postp=False,
        force_preprocess=True,
        min_nnz_genes=3,
        min_valid_genes_id=100,
    )
    adata = preprocessor(adata)

    if args.save and not args.dry_run:
        save_path = args.adata_path.replace(".h5ad", ".preprocessed.h5ad")
        adata.write_h5ad(save_path)
        print(f"Preprocessed dataset saved to {save_path}")
else:
    print(f"Dataset seems to be already preprocessed, skipping preprocessing step.")

if args.collection_name is None or args.dry_run:
    print("No collection name provided or dry run enabled, do not add to database.")
else:
    with FileLock("lamindb.lock"):
        try:
            collection = ln.Collection.filter(name=args.collection_name).one()

            if collection.artifacts.filter(description=args.adata_path).exists():
                print(
                    f"Dataset from path {args.adata_path} is already in collection {args.collection_name}. Skipping addition."
                )
            else:
                print(
                    f"Adding dataset from path {args.adata_path} to collection {args.collection_name}."
                )
                artifact = ln.Artifact.from_anndata(
                    adata,
                    description=args.adata_path,
                )
                artifact.save()
                collection.artifacts.add(artifact)
                collection.save()
                print(
                    f"Dataset {args.dataset_name} (from {args.adata_path}) successfully added to collection {args.collection_name}."
                )
                if collection.artifacts.filter(id=artifact.id).exists():
                    print("验证成功：Artifact 存在于 Collection 的 artifacts 列表中。")
                else:
                    print(
                        "验证失败：Artifact 未找到在 Collection 的 artifacts 列表中。"
                    )

        except ln.core.exceptions.DoesNotExist:
            print(
                f"Error: Collection '{args.collection_name}' does not exist. Init a new collection."
            )
            artifact = ln.Artifact.from_anndata(
                adata,
                description=args.adata_path,
            )
            artifact.save()
            collection = ln.Collection(
                [artifact],
                name=args.collection_name,
                description="Collection for single-cell datasets",
            )
            collection.save()
        except Exception as e:
            print(f"An error occurred during database operations: {e}")
