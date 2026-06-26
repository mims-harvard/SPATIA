import warnings
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from typing import Literal, Optional, Union
import os

import bionty as bt
import lamindb as ln
import lmdb
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from lamindb.core import MappedCollection
from lamindb.core._mapped_collection import _Connect
from lamindb.core.storage._anndata_accessor import _safer_read_index
from matplotlib.pylab import byte
from PIL import Image
from scdataloader.utils import get_ancestry_mapping, load_genes
from scipy.sparse import issparse
from torch.utils.data import Dataset as torchDataset
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)

from .config import LABELS_TOADD


@dataclass
class Dataset(torchDataset):

    lamin_dataset: ln.Collection
    genedf: Optional[pd.DataFrame] = None
    organisms: Optional[Union[list[str], str]] = field(
        default_factory=["NCBITaxon:9606", "NCBITaxon:10090"]
    )
    obs: Optional[list[str]] = field(
        default_factory=[
            "self_reported_ethnicity_ontology_term_id",
            "assay_ontology_term_id",
            "development_stage_ontology_term_id",
            "disease_ontology_term_id",
            "cell_type_ontology_term_id",
            "tissue_ontology_term_id",
            "sex_ontology_term_id",
        ]
    )
    spatial_datadir: Optional[Union[str, list[str]]] = None
    extra_obs: Optional[list[str]] = field(
        default_factory=[]
    )
    hierarchical_clss: Optional[list[str]] = field(default_factory=list)
    join_vars: Literal["inner", "outer"] | None = None
    clip_model_type: Optional[str] = "openai/clip-vit-base-patch32"

    def __post_init__(self):
        self.mapped_dataset = mapped(
            self.lamin_dataset,
            obs_keys=self.obs + self.extra_obs,
            join=self.join_vars,
            encode_labels=self.obs,
            unknown_label="unknown",
            stream=True,
            parallel=True,
        )
        print(
            "won't do any check but we recommend to have your dataset coming from local storage"
        )
        self.labels_groupings = {}
        self.class_topred = {}
        if len(self.hierarchical_clss) > 0:
            self.define_hierarchies(self.hierarchical_clss)
        if len(self.obs) > 0:
            for clss in self.obs:
                if clss not in self.hierarchical_clss:
                    self.class_topred[clss] = set(
                        self.mapped_dataset.get_merged_categories(clss)
                    )
                    if (
                        self.mapped_dataset.unknown_label
                        in self.mapped_dataset.encoders[clss].keys()
                    ):
                        self.class_topred[clss] -= set(
                            [self.mapped_dataset.unknown_label]
                        )

        if self.genedf is None:
            self.genedf = load_genes(self.organisms)

        self.genedf.columns = self.genedf.columns.astype(str)
        self.check_aligned_vars()
        from transformers import AutoImageProcessor
        self.image_preprocesser = AutoImageProcessor.from_pretrained(
            self.clip_model_type
        )

        self.env_list = []
        if self.spatial_datadir is not None:
            print(f"初始化LMDB，传入的空间数据路径: {self.spatial_datadir}")
            
            if isinstance(self.spatial_datadir, str):
                lmdb_paths = [path.strip() for path in self.spatial_datadir.split(",")]
            elif isinstance(self.spatial_datadir, list):
                lmdb_paths = self.spatial_datadir
            else:
                raise TypeError(f"spatial_datadir must be a string or list, got {type(self.spatial_datadir)}")
            
            print(f"LMDB路径列表: {lmdb_paths}")
            
            for i, lmdb_path in enumerate(lmdb_paths):
                try:
                    print(f"尝试打开LMDB {i+1}/{len(lmdb_paths)}: {lmdb_path}")
                    env = lmdb.open(
                        lmdb_path,
                        readonly=True,
                        lock=False,
                        subdir=False,
                        map_size=1024**4 * 4,
                    )
                    self.env_list.append(env)
                    print(f"成功打开LMDB: {lmdb_path}")
                except Exception as e:
                    print(f"打开LMDB文件失败 {lmdb_path}: {str(e)}")
            
            print(f"共成功打开 {len(self.env_list)} 个LMDB环境")
            
            if not self.env_list:
                print("警告: 没有有效的LMDB环境被打开，将spatial_datadir设置为None")
                self.spatial_datadir = None

    def check_aligned_vars(self):
        vars = self.genedf.index.tolist()
        i = 0
        for storage in self.mapped_dataset.storages:
            with _Connect(storage) as store:
                if len(set(_safer_read_index(store["var"]).tolist()) - set(vars)) == 0:
                    i += 1
        print("{}% are aligned".format(i * 100 / len(self.mapped_dataset.storages)))

    def __len__(self, **kwargs):
        return self.mapped_dataset.__len__(**kwargs)

    @property
    def encoder(self):
        return self.mapped_dataset.encoders

    def __getitem__(self, *args, **kwargs):
        item = self.mapped_dataset.__getitem__(*args, **kwargs)
        if self.spatial_datadir is not None and self.env_list:
            dataset_name = item["dataset_name"].split("/")[-1]
            cell_id = item["index"]
            if isinstance(cell_id, bytes):
                cell_id = cell_id.decode("utf-8")
            key = f"{dataset_name}/{cell_id}".encode("utf-8")
            
            for i, env in enumerate(self.env_list):
                image_key = "image"
                if i == 1:
                    image_key = "region_image"
                elif i == 2:
                    image_key = "tissue_image"
                
                with env.begin() as txn:
                    try:
                        data = np.frombuffer(txn.get(key), dtype=np.uint8).reshape(256, 256)
                        image = self.image_preprocesser(
                            images=np.stack((data,) * 3, axis=-1), return_tensors="pt"
                        )["pixel_values"][0]
                        item[image_key] = image
                    except Exception as e:
                        item[image_key] = torch.zeros((3, 224, 224))
        
        return item

    def __repr__(self):
        return (
            "total dataset size is {} Gb\n".format(
                sum([file.size for file in self.lamin_dataset.artifacts.all()]) / 1e9
            )
            + "---\n"
            + "dataset contains:\n"
            + "     {} cells\n".format(self.mapped_dataset.__len__())
            + "     {} genes\n".format(self.genedf.shape[0])
            + "     {} labels\n".format(len(self.obs))
            + "     {} obs\n".format(len(self.obs))
            + "     {} hierarchical_clss\n".format(len(self.hierarchical_clss))
            + "     {} organisms\n".format(len(self.organisms))
            + (
                "dataset contains {} classes to predict\n".format(
                    sum([len(self.class_topred[i]) for i in self.class_topred])
                )
                if len(self.class_topred) > 0
                else ""
            )
        )

    def get_label_weights(self, obs_keys: str | list[str], scaler: int = 10):
        if isinstance(obs_keys, str):
            obs_keys = [obs_keys]
        labels_list = []
        for label_key in obs_keys:
            labels_to_str = (
                self.mapped_dataset.get_merged_labels(label_key).astype(str).astype("O")
            )
            labels_list.append(labels_to_str)
        if len(labels_list) > 1:
            labels = reduce(lambda a, b: a + b, labels_list)
        else:
            labels = labels_list[0]

        counter = Counter(labels)
        rn = {n: i for i, n in enumerate(counter.keys())}
        labels = np.array([rn[label] for label in labels])
        counter = np.array(list(counter.values()))
        weights = scaler / (counter + scaler)
        return weights, labels

    def get_unseen_mapped_dataset_elements(self, idx: int):
        return [str(i)[2:-1] for i in self.mapped_dataset.uns(idx, "unseen_genes")]

    def define_hierarchies(self, clsses: list[str]):
        self.labels_groupings = {}
        self.class_topred = {}
        for clss in clsses:
            if clss not in [
                "cell_type_ontology_term_id",
                "tissue_ontology_term_id",
                "disease_ontology_term_id",
                "development_stage_ontology_term_id",
                "assay_ontology_term_id",
                "self_reported_ethnicity_ontology_term_id",
            ]:
                raise ValueError(
                    "class {} not in accepted classes, for now only supported from bionty sources".format(
                        clss
                    )
                )
            elif clss == "cell_type_ontology_term_id":
                parentdf = (
                    bt.CellType.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "tissue_ontology_term_id":
                parentdf = (
                    bt.Tissue.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "disease_ontology_term_id":
                parentdf = (
                    bt.Disease.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "development_stage_ontology_term_id":
                parentdf = (
                    bt.DevelopmentalStage.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "assay_ontology_term_id":
                parentdf = (
                    bt.ExperimentalFactor.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )
            elif clss == "self_reported_ethnicity_ontology_term_id":
                parentdf = (
                    bt.Ethnicity.filter()
                    .df(include=["parents__ontology_id"])
                    .set_index("ontology_id")
                )

            else:
                raise ValueError(
                    "class {} not in accepted classes, for now only supported from bionty sources".format(
                        clss
                    )
                )
            cats = set(self.mapped_dataset.get_merged_categories(clss))
            addition = set(LABELS_TOADD.get(clss, {}).values())
            cats |= addition
            groupings, _, leaf_labels = get_ancestry_mapping(cats, parentdf)
            for i, j in groupings.items():
                if len(j) == 0:
                    groupings.pop(i)
            self.labels_groupings[clss] = groupings
            if clss in self.obs:
                mlength = len(self.mapped_dataset.encoders[clss])

                mlength -= (
                    1
                    if self.mapped_dataset.unknown_label
                    in self.mapped_dataset.encoders[clss].keys()
                    else 0
                )

                for i, v in enumerate(
                    addition - set(self.mapped_dataset.encoders[clss].keys())
                ):
                    self.mapped_dataset.encoders[clss].update({v: mlength + i})

                self.class_topred[clss] = leaf_labels
                c = 0
                update = {}
                mlength = len(leaf_labels)
                mlength -= (
                    1
                    if self.mapped_dataset.unknown_label
                    in self.mapped_dataset.encoders[clss].keys()
                    else 0
                )
                for k, v in self.mapped_dataset.encoders[clss].items():
                    if k in self.labels_groupings[clss].keys():
                        update.update({k: mlength + c})
                        c += 1
                    elif k == self.mapped_dataset.unknown_label:
                        update.update({k: v})
                        self.class_topred[clss] -= set([k])
                    else:
                        update.update({k: v - c})
                self.mapped_dataset.encoders[clss] = update


class SimpleAnnDataset(torchDataset):
    def __init__(
        self,
        adata: AnnData,
        obs_to_output: Optional[list[str]] = [],
        layer: Optional[str] = None,
        spatial_datadir: Optional[str] = None,
        image_preprocesser: Optional[str] = None,
        fix_missing_image: bool = False,
    ):
        self.adataX = adata.layers[layer] if layer is not None else adata.X
        self.adataX = self.adataX.toarray() if issparse(self.adataX) else self.adataX
        self.obs_to_output = adata.obs[obs_to_output]
        self.fix_missing_image = fix_missing_image

        self.obs_index = adata.obs.index.astype(str).tolist()
        self.dataset_name = (
            adata.obs["dataset_name"].astype(str).tolist()
            if "dataset_name" in adata.obs.columns
            else [None] * len(self.obs_index)
        )

        self.env_list = []
        self.spatial_datadir = spatial_datadir
        self.image_preprocesser_obj = None

        if spatial_datadir is not None:
            from transformers import AutoImageProcessor
            model_name = image_preprocesser or "facebook/vit-mae-base"
            self.image_preprocesser_obj = AutoImageProcessor.from_pretrained(model_name)

            if isinstance(spatial_datadir, str):
                lmdb_paths = [p.strip() for p in spatial_datadir.split(",")]
            elif isinstance(spatial_datadir, list):
                lmdb_paths = spatial_datadir
            else:
                lmdb_paths = [spatial_datadir]

            for lmdb_path in lmdb_paths:
                try:
                    env = lmdb.open(
                        lmdb_path,
                        readonly=True,
                        lock=False,
                        subdir=False,
                        map_size=1024**4 * 4,
                    )
                    self.env_list.append(env)
                except Exception as e:
                    print(f"Warning: failed to open LMDB {lmdb_path}: {e}")

            if not self.env_list:
                print("Warning: no valid LMDB environments opened, disabling spatial")
                self.spatial_datadir = None

    def __len__(self):
        return self.adataX.shape[0]

    def _load_image(self, idx):
        ds_name = self.dataset_name[idx]
        cell_id = self.obs_index[idx]

        if ds_name is not None:
            ds_name = ds_name.split("/")[-1]
            key = f"{ds_name}/{cell_id}".encode("utf-8")
        else:
            key = cell_id.encode("utf-8")

        for env in self.env_list:
            with env.begin() as txn:
                data = txn.get(key)
                if data is not None:
                    arr = np.frombuffer(data, dtype=np.uint8).reshape(256, 256)
                    rgb = np.stack((arr,) * 3, axis=-1)
                    image = self.image_preprocesser_obj(
                        images=rgb, return_tensors="pt"
                    )["pixel_values"][0]
                    return image

        if self.fix_missing_image:
            return torch.zeros((3, 224, 224))
        raise KeyError(f"Image not found in any LMDB for key: {key.decode()}")

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            out = {"X": self.adataX[idx].reshape(-1)}
            out.update(
                {name: val for name, val in self.obs_to_output.iloc[idx].items()}
            )
        if self.spatial_datadir is not None and self.env_list:
            out["image"] = self._load_image(idx)
        return out


def mapped(
    dataset,
    obs_keys: list[str] | None = None,
    join: Literal["inner", "outer"] | None = "inner",
    encode_labels: bool | list[str] = True,
    unknown_label: str | dict[str, str] | None = None,
    cache_categories: bool = True,
    parallel: bool = False,
    dtype: str | None = None,
    stream: bool = False,
    is_run_input: bool | None = None,
) -> MappedCollection:
    path_list = []
    for artifact in dataset.artifacts.all():
        if artifact.suffix not in {".h5ad", ".zrad", ".zarr"}:
            print(f"Ignoring artifact with suffix {artifact.suffix}")
            continue
        elif not artifact.path.exists():
            print(f"Path does not exist for artifact with suffix {artifact.suffix}")
            continue
        elif not stream:
            path_list.append(artifact.stage())
        else:
            path_list.append(artifact.path)
    ds = MappedCollection(
        path_list=path_list,
        obs_keys=obs_keys,
        join=join,
        encode_labels=encode_labels,
        unknown_label=unknown_label,
        cache_categories=cache_categories,
        parallel=parallel,
        dtype=dtype,
    )
    return ds
