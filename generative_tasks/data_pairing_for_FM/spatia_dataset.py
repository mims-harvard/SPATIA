
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import lmdb
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings


class SpatiaImageTransforms:
    
    @staticmethod
    def get_train_transforms(image_size: Tuple[int, int] = (256, 256)) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    @staticmethod
    def get_val_transforms(image_size: Tuple[int, int] = (256, 256)) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def _normalize_dataset_name(name: str) -> str:
    base = str(name).split("/")[-1]
    if base.endswith(".h5ad"):
        base = base[:-5]
    return base


def _open_lmdb_robust(lmdb_path: str) -> lmdb.Environment:
    try:
        return lmdb.open(
            lmdb_path, 
            readonly=True, 
            lock=False, 
            subdir=False, 
            map_size=1024**4 * 4
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open LMDB at {lmdb_path}: {e}")


def _build_image_key_for_index(adata: ad.AnnData, idx: int, adata_path: Path) -> str:
    if "dataset_name" in adata.obs.columns:
        dataset_series = adata.obs.loc[adata.obs.index[idx], "dataset_name"]
        name = _normalize_dataset_name(str(dataset_series))
    else:
        name = _normalize_dataset_name(adata_path.stem)
    
    if "cell_id" in adata.obs.columns:
        cell_id = str(adata.obs.loc[adata.obs.index[idx], "cell_id"])
    elif "index" in adata.obs.columns:
        cell_id = str(adata.obs.loc[adata.obs.index[idx], "index"])
    else:
        cell_id = str(adata.obs.index[idx])
    
    return f"{name}/{cell_id}"


def _read_image_from_lmdb_robust(env: lmdb.Environment, key: str, fix_missing: bool = True) -> np.ndarray:
    with env.begin() as txn:
        raw = None
        meta_bytes = None
        
        for key_pattern in [key, f"{key}:image", f"{key}/image"]:
            raw = txn.get(key_pattern.encode("utf-8"))
            if raw is not None:
                for meta_pattern in [f"{key}_meta", f"{key}:meta", f"{key}/meta"]:
                    meta_bytes = txn.get(meta_pattern.encode("utf-8"))
                    if meta_bytes is not None:
                        break
                break
        
        if raw is None:
            if fix_missing:
                warnings.warn(f"Image for key {key} not found, using placeholder")
                return np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                raise KeyError(f"Image for key {key} not found in LMDB")
        
        meta = {}
        if meta_bytes:
            try:
                meta = json.loads(meta_bytes.decode("utf-8"))
            except:
                warnings.warn(f"Could not parse metadata for key {key}")
        
        fmt = meta.get("format", "png").lower()
        try:
            if fmt == "png":
                arr = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            else:
                dtype = np.dtype(meta.get("dtype", "uint8"))
                shape = meta.get("shape")
                data = np.frombuffer(raw, dtype=dtype)
                if shape:
                    arr = data.reshape(shape)
                else:
                    side = int(np.sqrt(data.size))
                    arr = data.reshape(side, side)
                if arr.ndim == 2:
                    arr = np.stack((arr,) * 3, axis=-1)
        except Exception as e:
            if fix_missing:
                warnings.warn(f"Could not decode image for key {key}: {e}, using placeholder")
                return np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                raise RuntimeError(f"Could not decode image for key {key}: {e}")
        
        return arr.astype(np.uint8)


class SpatiaPairedDataset(Dataset):
    
    def __init__(
        self,
        adata_path: str,
        pairs_csv: str,
        lmdb_path: str,
        expr_layer: Optional[str] = None,
        return_expr: bool = True,
        delta_g_npz: Optional[str] = None,
        delta_m_npz: Optional[str] = None,
        fix_missing_images: bool = True,
        image_size: Optional[Tuple[int, int]] = (256, 256),
        transforms: Optional[transforms.Compose] = None,
        normalize_expr: bool = True,
        log_transform_expr: bool = False,
        max_expr_value: float = 10.0,
    ):
        super().__init__()
        
        self.adata_path = Path(adata_path)
        self.expr_layer = expr_layer
        self.return_expr = return_expr
        self.fix_missing_images = fix_missing_images
        self.image_size = image_size
        self.transforms = transforms
        self.normalize_expr = normalize_expr
        self.log_transform_expr = log_transform_expr
        self.max_expr_value = max_expr_value
        
        print(f"Loading AnnData from {self.adata_path}")
        self.adata = sc.read_h5ad(self.adata_path)
        print(f"AnnData shape: {self.adata.shape}")
        
        print(f"Loading pairs from {pairs_csv}")
        self.pairs = pd.read_csv(pairs_csv)
        print(f"Found {len(self.pairs)} pairs")
        
        valid_pairs = []
        for i, row in self.pairs.iterrows():
            idx_c = int(row.get("x_ctrl_id", row.get("control_index", -1)))
            idx_t = int(row.get("x_tgt_id", row.get("perturbation_index", -1)))
            
            if 0 <= idx_c < len(self.adata) and 0 <= idx_t < len(self.adata):
                valid_pairs.append(i)
            else:
                warnings.warn(f"Invalid indices in pair {i}: ctrl={idx_c}, tgt={idx_t}")
        
        self.pairs = self.pairs.iloc[valid_pairs].reset_index(drop=True)
        print(f"Validated {len(self.pairs)} pairs with valid indices")
        
        print(f"Opening LMDB at {lmdb_path}")
        self.env = _open_lmdb_robust(lmdb_path)
        
        self.delta_g_map: Dict[str, np.ndarray] = {}
        if delta_g_npz is not None and os.path.exists(delta_g_npz):
            print(f"Loading Δg signatures from {delta_g_npz}")
            try:
                npz = np.load(delta_g_npz, allow_pickle=True)
                keys = npz["delta_g_keys"].tolist()
                vals = npz["delta_g_values"].tolist()
                self.delta_g_map = {k: v for k, v in zip(keys, vals)}
                print(f"Loaded {len(self.delta_g_map)} Δg signatures")
            except Exception as e:
                warnings.warn(f"Could not load Δg signatures: {e}")
        
        self.delta_m_map: Dict[str, np.ndarray] = {}
        if delta_m_npz is not None and os.path.exists(delta_m_npz):
            print(f"Loading Δm signatures from {delta_m_npz}")
            try:
                npz = np.load(delta_m_npz, allow_pickle=True)
                for key in npz.files:
                    self.delta_m_map[key] = npz[key]
                print(f"Loaded {len(self.delta_m_map)} Δm signatures")
                if self.delta_m_map:
                    sample_key = list(self.delta_m_map.keys())[0]
                    print(f"  Sample Δm shape: {self.delta_m_map[sample_key].shape}")
            except Exception as e:
                warnings.warn(f"Could not load Δm signatures: {e}")
        
        if self.transforms is None:
            if self.image_size is not None:
                self.transforms = SpatiaImageTransforms.get_val_transforms(self.image_size)
        
        if self.return_expr and self.normalize_expr:
            self._compute_expr_stats()
    
    def _compute_expr_stats(self):
        print("Computing expression statistics...")
        X = self.adata.layers[self.expr_layer] if self.expr_layer else self.adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        
        self.expr_mean = np.mean(X, axis=0)
        self.expr_std = np.std(X, axis=0)
        self.expr_std[self.expr_std == 0] = 1.0
        
        print(f"Expression statistics computed: mean range [{self.expr_mean.min():.3f}, {self.expr_mean.max():.3f}]")
    
    def _get_expression(self, idx: int) -> np.ndarray:
        X = self.adata.layers[self.expr_layer][idx] if self.expr_layer else self.adata.X[idx]
        if hasattr(X, "toarray"):
            X = X.toarray()
        expr = np.asarray(X).flatten().astype(np.float32)
        
        if self.log_transform_expr:
            expr = np.log1p(expr)
        
        if self.normalize_expr:
            expr = (expr - self.expr_mean) / self.expr_std
            expr = np.clip(expr, -self.max_expr_value, self.max_expr_value)
        
        return expr
    
    def _get_image(self, idx: int) -> torch.Tensor:
        key = _build_image_key_for_index(self.adata, idx, self.adata_path)
        arr = _read_image_from_lmdb_robust(self.env, key, self.fix_missing_images)
        
        if self.transforms is not None:
            if isinstance(arr, np.ndarray):
                if arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
                image_tensor = self.transforms(arr)
            else:
                image_tensor = self.transforms(arr)
        else:
            if self.image_size is not None:
                arr = np.array(Image.fromarray(arr).resize(self.image_size, Image.BILINEAR))
            if arr.ndim == 2:
                arr = np.stack((arr,) * 3, axis=-1)
            arr = arr.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        
        return image_tensor
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, i: int) -> Dict[str, Union[torch.Tensor, str, int, float]]:
        row = self.pairs.iloc[i]
        
        idx_c = int(row.get("x_ctrl_id", row.get("control_index", -1)))
        idx_t = int(row.get("x_tgt_id", row.get("perturbation_index", -1)))
        
        ctrl_img = self._get_image(idx_c)
        tgt_img = self._get_image(idx_t)
        
        output = {
            "ctrl_image": ctrl_img,
            "tgt_image": tgt_img,
            "ctrl_idx": idx_c,
            "tgt_idx": idx_t,
            "state_A": row.get("state_A", "unknown"),
            "state_B": row.get("state_B", "unknown"),
            "cell_type": row.get("cell_type", "unknown"),
            "niche": row.get("niche", "unknown"),
            "transition_tag": row.get("transition_tag", "unknown"),
        }
        
        output["ot_confidence"] = float(row.get("ot_confidence", 1.0))
        
        if self.return_expr:
            output["ctrl_expr"] = torch.from_numpy(self._get_expression(idx_c))
            output["tgt_expr"] = torch.from_numpy(self._get_expression(idx_t))
        
        transition_tag = row.get("transition_tag")
        cell_type = row.get("cell_type")
        niche = row.get("niche") 
        state_A = row.get("state_A")
        state_B = row.get("state_B")
        
        if self.delta_g_map:
            
            delta_g = None
            for key in [
                transition_tag,
                f"{cell_type}|{niche}|{state_A}->{state_B}",
            ]:
                if key and key in self.delta_g_map:
                    delta_g = self.delta_g_map[key]
                    break
            
            if delta_g is not None:
                if isinstance(delta_g, list):
                    delta_g = np.array(delta_g, dtype=np.float32)
                elif hasattr(delta_g, 'astype'):
                    delta_g = delta_g.astype(np.float32)
                else:
                    delta_g = np.array(delta_g, dtype=np.float32)
                
                if self.normalize_expr and delta_g.shape[0] == self.expr_std.shape[0]:
                    delta_g = delta_g / self.expr_std
                    delta_g = np.clip(delta_g, -self.max_expr_value, self.max_expr_value)
                output["delta_g"] = torch.from_numpy(delta_g)
        
        if self.delta_m_map:
            delta_m = None
            if transition_tag and transition_tag in self.delta_m_map:
                delta_m = self.delta_m_map[transition_tag]
            
            if delta_m is not None:
                if isinstance(delta_m, list):
                    delta_m = np.array(delta_m, dtype=np.float32)
                elif hasattr(delta_m, 'astype'):
                    delta_m = delta_m.astype(np.float32)
                else:
                    delta_m = np.array(delta_m, dtype=np.float32)
                
                delta_m_mean = np.array([1000, 100, 0.5, 0.8, 128, 200, 50, 40, 50, 20], dtype=np.float32)
                delta_m_std = np.array([2000, 200, 0.3, 0.2, 50, 50, 50, 30, 50, 30], dtype=np.float32)
                delta_m_std[delta_m_std == 0] = 1.0
                delta_m = (delta_m) / (delta_m_std + 1e-8)
                delta_m = np.clip(delta_m, -3.0, 3.0)
                
                output["delta_m"] = torch.from_numpy(delta_m)
            else:
                output["delta_m"] = torch.zeros(10, dtype=torch.float32)
        
        return output
    
    def get_data_stats(self) -> Dict[str, Union[int, float, List]]:
        stats = {
            "n_pairs": len(self),
            "n_cells": len(self.adata),
            "n_genes": self.adata.n_vars,
            "image_size": self.image_size,
            "has_delta_g": len(self.delta_g_map) > 0,
            "n_delta_g": len(self.delta_g_map),
            "has_delta_m": len(self.delta_m_map) > 0,
            "n_delta_m": len(self.delta_m_map),
            "delta_m_dim": 10,
        }
        
        if len(self.pairs) > 0:
            stats.update({
                "states_A": self.pairs["state_A"].value_counts().to_dict() if "state_A" in self.pairs.columns else {},
                "states_B": self.pairs["state_B"].value_counts().to_dict() if "state_B" in self.pairs.columns else {},
                "cell_types": self.pairs["cell_type"].value_counts().to_dict() if "cell_type" in self.pairs.columns else {},
                "niches": self.pairs["niche"].value_counts().to_dict() if "niche" in self.pairs.columns else {},
            })
            
            if "ot_confidence" in self.pairs.columns:
                conf = self.pairs["ot_confidence"]
                stats.update({
                    "ot_confidence_mean": float(conf.mean()),
                    "ot_confidence_std": float(conf.std()),
                    "ot_confidence_min": float(conf.min()),
                    "ot_confidence_max": float(conf.max()),
                })
        
        return stats


def create_spatia_dataloaders(
    dataset: SpatiaPairedDataset,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    
    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        return train_loader, None


if __name__ == "__main__":
    dataset = SpatiaPairedDataset(
        adata_path="./data/adata_processed_with_cell_state.h5ad",
        pairs_csv="perturbation_pairs.csv",
        lmdb_path="./data/lmdb/xenium_he.lmdb",
        return_expr=True,
        fix_missing_images=True,
        image_size=(256, 256)
    )
    
    print(f"Dataset created with {len(dataset)} pairs")
    print("Dataset stats:", dataset.get_data_stats())
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", list(sample.keys()))
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {value}")
