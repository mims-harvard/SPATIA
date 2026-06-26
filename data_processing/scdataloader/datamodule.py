from typing import Optional, Sequence, Union

import ipdb
import lamindb as ln
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import (
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

from .collator import Collator
from .data import Dataset
from .utils import getBiomartTable


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        collection_name: str,
        clss_to_weight: list = ["organism_ontology_term_id"],
        organisms: list = ["NCBITaxon:9606"],
        weight_scaler: int = 10,
        train_oversampling_per_epoch: float = 0.1,
        validation_split: float = 0.2,
        test_split: float = 0,
        gene_embeddings: str = "",
        use_default_col: bool = True,
        gene_position_tolerance: int = 10_000,
        all_clss: list = ["organism_ontology_term_id"],
        hierarchical_clss: list = [],
        how: str = "random expr",
        organism_name: str = "organism_ontology_term_id",
        max_len: int = 1000,
        add_zero_genes: int = 100,
        do_gene_pos: Union[bool, str] = True,
        tp_name: Optional[str] = None,
        assays_to_drop: list = [
            "EFO:0008853",
            "EFO:0010961",
            "EFO:0030007",
            "EFO:0030062",
        ],
        **kwargs,
    ):
        if collection_name is not None:
            mdataset = Dataset(
                ln.Collection.filter(name=collection_name).first(),
                organisms=organisms,
                obs=all_clss,
                hierarchical_clss=hierarchical_clss,
            )
        self.gene_pos = None
        if do_gene_pos:
            if type(do_gene_pos) is str:
                print("seeing a string: loading gene positions as biomart parquet file")
                biomart = pd.read_parquet(do_gene_pos)
            else:
                if organisms != ["NCBITaxon:9606"]:
                    raise ValueError(
                        "need to provide your own table as this automated function only works for humans for now"
                    )
                biomart = getBiomartTable(
                    attributes=["start_position", "chromosome_name"],
                    useCache=True,
                ).set_index("ensembl_gene_id")
                biomart = biomart.loc[~biomart.index.duplicated(keep="first")]
                biomart = biomart.sort_values(by=["chromosome_name", "start_position"])
                c = []
                i = 0
                prev_position = -100000
                prev_chromosome = None
                for _, r in biomart.iterrows():
                    if (
                        r["chromosome_name"] != prev_chromosome
                        or r["start_position"] - prev_position > gene_position_tolerance
                    ):
                        i += 1
                    c.append(i)
                    prev_position = r["start_position"]
                    prev_chromosome = r["chromosome_name"]
                print(f"reduced the size to {len(set(c))/len(biomart)}")
                biomart["pos"] = c
            mdataset.genedf = mdataset.genedf.join(biomart, how="inner")
            self.gene_pos = mdataset.genedf["pos"].astype(int).tolist()

        if gene_embeddings != "":
            mdataset.genedf = mdataset.genedf.join(
                pd.read_parquet(gene_embeddings), how="inner"
            )
            if do_gene_pos:
                self.gene_pos = mdataset.genedf["pos"].tolist()
        self.classes = {k: len(v) for k, v in mdataset.class_topred.items()}
        if use_default_col:
            kwargs["collate_fn"] = Collator(
                organisms=organisms,
                how=how,
                valid_genes=mdataset.genedf.index.tolist(),
                max_len=max_len,
                add_zero_genes=add_zero_genes,
                org_to_id=mdataset.encoder[organism_name],
                tp_name=tp_name,
                organism_name=organism_name,
                class_names=clss_to_weight,
            )
        self.validation_split = validation_split
        self.test_split = test_split
        self.dataset = mdataset
        self.kwargs = kwargs
        if "sampler" in self.kwargs:
            self.kwargs.pop("sampler")
        self.assays_to_drop = assays_to_drop
        self.n_samples = len(mdataset)
        self.weight_scaler = weight_scaler
        self.train_oversampling_per_epoch = train_oversampling_per_epoch
        self.clss_to_weight = clss_to_weight
        self.train_weights = None
        self.train_labels = None
        self.test_datasets = []
        self.test_idx = []
        super().__init__()

    def __repr__(self):
        return (
            f"DataLoader(\n"
            f"\twith a dataset=({self.dataset.__repr__()}\n)\n"
            f"\tvalidation_split={self.validation_split},\n"
            f"\ttest_split={self.test_split},\n"
            f"\tn_samples={self.n_samples},\n"
            f"\tweight_scaler={self.weight_scaler},\n"
            f"\ttrain_oversampling_per_epoch={self.train_oversampling_per_epoch},\n"
            f"\tassays_to_drop={self.assays_to_drop},\n"
            f"\tnum_datasets={len(self.dataset.mapped_dataset.storages)},\n"
            f"\ttest datasets={str(self.test_datasets)},\n"
            f"perc test: {str(len(self.test_idx) / self.n_samples)},\n"
            f"\tclss_to_weight={self.clss_to_weight}\n"
            + (
                (
                    "\twith train_dataset size of=("
                    + str((self.train_weights != 0).sum())
                    + ")\n)"
                )
                if self.train_weights is not None
                else ")"
            )
        )

    @property
    def decoders(self):
        decoders = {}
        for k, v in self.dataset.encoder.items():
            decoders[k] = {va: ke for ke, va in v.items()}
        return decoders

    @property
    def labels_hierarchy(self):
        labels_hierarchy = {}
        for k, dic in self.dataset.labels_groupings.items():
            rdic = {}
            for sk, v in dic.items():
                rdic[self.dataset.encoder[k][sk]] = [
                    self.dataset.encoder[k][i] for i in list(v)
                ]
            labels_hierarchy[k] = rdic
        return labels_hierarchy

    @property
    def genes(self):
        return self.dataset.genedf.index.tolist()

    @property
    def num_datasets(self):
        return len(self.dataset.mapped_dataset.storages)

    def setup(self, stage=None):
        if len(self.clss_to_weight) > 0 and self.weight_scaler > 0:
            weights, labels = self.dataset.get_label_weights(
                self.clss_to_weight, scaler=self.weight_scaler
            )
        else:
            weights = np.ones(1)
            labels = np.zeros(self.n_samples)
        if isinstance(self.validation_split, int):
            len_valid = self.validation_split
        else:
            len_valid = int(self.n_samples * self.validation_split)
        if isinstance(self.test_split, int):
            len_test = self.test_split
        else:
            len_test = int(self.n_samples * self.test_split)
        assert (
            len_test + len_valid < self.n_samples
        ), "test set + valid set size is configured to be larger than entire dataset."

        idx_full = []
        if len(self.assays_to_drop) > 0:
            for i, a in enumerate(
                self.dataset.mapped_dataset.get_merged_labels("assay_ontology_term_id")
            ):
                if a not in self.assays_to_drop:
                    idx_full.append(i)
            idx_full = np.array(idx_full)
        else:
            idx_full = np.arange(self.n_samples)

        if len_test > 0:
            self.test_idx = idx_full[:len_test]
            idx_full = idx_full[len_test:]
        else:
            self.test_idx = None


        np.random.shuffle(idx_full)
        if len_valid > 0:
            self.valid_idx = idx_full[:len_valid].copy()
            idx_full = idx_full[len_valid:]
        else:
            self.valid_idx = None
        weights = np.concatenate([weights, np.zeros(1)])
        labels[~np.isin(np.arange(self.n_samples), idx_full)] = len(weights) - 1

        self.train_weights = weights
        self.train_labels = labels
        self.idx_full = idx_full
        return self.test_datasets

    def train_dataloader(self, **kwargs):
        print(f"valid_idx: {self.valid_idx}")
        print(f"idx_full: {self.idx_full}")
        return DataLoader(
            self.dataset, sampler=SubsetRandomSampler(self.idx_full), **self.kwargs
        )

        print(f"train weights: {self.train_weights}, train labels: {self.train_labels}")
        print(f"self.kwarg: {self.kwargs}, kwargs: {kwargs}")
        train_sampler = LabelWeightedSampler(
            self.train_weights,
            self.train_labels,
            num_samples=int(self.n_samples * self.train_oversampling_per_epoch),
        )
        return DataLoader(self.dataset, sampler=train_sampler, **self.kwargs, **kwargs)

    def val_dataloader(self):
        return (
            DataLoader(
                self.dataset, sampler=SubsetRandomSampler(self.valid_idx), **self.kwargs
            )
            if self.valid_idx is not None
            else None
        )

    def test_dataloader(self):
        return (
            DataLoader(
                self.dataset, sampler=SequentialSampler(self.test_idx), **self.kwargs
            )
            if self.test_idx is not None
            else None
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset, sampler=SubsetRandomSampler(self.idx_full), **self.kwargs
        )


class LabelWeightedSampler(Sampler[int]):
    label_weights: Sequence[float]
    klass_indices: Sequence[Sequence[int]]
    num_samples: int

    def __init__(
        self, label_weights: Sequence[float], labels: Sequence[int], num_samples: int
    ) -> None:

        super(LabelWeightedSampler, self).__init__(None)
        label_weights = np.array(label_weights) * np.bincount(labels)

        self.label_weights = torch.as_tensor(label_weights, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.int)
        self.num_samples = num_samples
        self.klass_indices = [
            (self.labels == i_klass).nonzero().squeeze(1)
            for i_klass in range(len(label_weights))
        ]

    def __iter__(self):
        sample_labels = torch.multinomial(
            self.label_weights, num_samples=self.num_samples, replacement=True
        )
        sample_indices = torch.empty_like(sample_labels)
        for i_klass, klass_index in enumerate(self.klass_indices):
            if klass_index.numel() == 0:
                continue
            left_inds = (sample_labels == i_klass).nonzero().squeeze(1)
            right_inds = torch.randint(len(klass_index), size=(len(left_inds),))
            sample_indices[left_inds] = klass_index[right_inds]
        yield from iter(sample_indices.tolist())

    def __len__(self):
        return self.num_samples
