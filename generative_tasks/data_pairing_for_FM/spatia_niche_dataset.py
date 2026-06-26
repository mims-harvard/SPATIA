
import io
import json
import lmdb
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Optional, Tuple, Union
import warnings


class NicheImageTransforms:
    
    @staticmethod
    def get_train_transforms(image_size: Tuple[int, int] = (512, 512)) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms(image_size: Tuple[int, int] = (512, 512)) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class SpatiaNicheDataset(Dataset):
    
    def __init__(
        self,
        adata_path: str,
        pairs_csv: str,
        wsi_image_path: Optional[str] = None,
        delta_g_npz: Optional[str] = None,
        lmdb_path: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        return_expr: bool = True,
        pixel_per_um: float = 1.0,
        transforms: Optional[transforms.Compose] = None,
        normalize_expr: bool = True,
        verbose: bool = True,
        dataset_name: str = "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs",
    ):
        super().__init__()
        
        self.adata_path = Path(adata_path)
        self.wsi_image_path = Path(wsi_image_path) if wsi_image_path else None
        self.lmdb_path = lmdb_path
        self.image_size = image_size
        self.return_expr = return_expr
        self.pixel_per_um = pixel_per_um
        self.transforms = transforms
        self.normalize_expr = normalize_expr
        self.verbose = verbose
        self.dataset_name = dataset_name
        
        if verbose:
            print(f"Loading niche pairs from {pairs_csv}")
        self.pairs = pd.read_csv(pairs_csv)
        if verbose:
            print(f"Found {len(self.pairs)} niche pairs")
        
        if return_expr:
            if verbose:
                print(f"Loading AnnData from {adata_path}")
            self.adata = sc.read_h5ad(adata_path)
            self._compute_niche_expressions()
        else:
            self.adata = None
        
        self.delta_g_map = {}
        if delta_g_npz and Path(delta_g_npz).exists():
            if verbose:
                print(f"Loading Δg from {delta_g_npz}")
            npz = np.load(delta_g_npz, allow_pickle=True)
            keys = npz['delta_g_keys'].tolist()
            vals = npz['delta_g_values'].tolist()
            self.delta_g_map = {k: np.asarray(v, dtype=np.float32) for k, v in zip(keys, vals)}
            if verbose:
                print(f"Loaded {len(self.delta_g_map)} Δg signatures")
        
        self.lmdb_env = None
        if lmdb_path and Path(lmdb_path).exists():
            if verbose:
                print(f"Opening LMDB at {lmdb_path}")
            self.lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        
        self.wsi_image = None
        if self.wsi_image_path and self.wsi_image_path.exists() and not self.lmdb_env:
            if verbose:
                print(f"Loading WSI from {wsi_image_path}")
            try:
                self.wsi_image = np.array(Image.open(self.wsi_image_path))
                if verbose:
                    print(f"WSI shape: {self.wsi_image.shape}")
            except Exception as e:
                warnings.warn(f"Could not load WSI: {e}")
        
        if self.transforms is None:
            self.transforms = NicheImageTransforms.get_val_transforms(image_size)
    
    def _compute_niche_expressions(self):
        if self.verbose:
            print("Computing pooled expressions per niche...")
        
        self.niche_expressions = {}
        niche_col = 'niche' if 'niche' in self.adata.obs.columns else self.pairs['source_niche_id'].name
        
        all_niches = set(self.pairs['source_niche_id'].tolist() + self.pairs['target_niche_id'].tolist())
        
        for niche_id in all_niches:
            mask = self.adata.obs[niche_col] == niche_id
            if mask.sum() > 0:
                X = self.adata.X[mask]
                if hasattr(X, 'toarray'):
                    X = X.toarray()
                pooled = np.mean(X, axis=0).astype(np.float32).flatten()
                self.niche_expressions[niche_id] = pooled
        
        if self.normalize_expr and self.niche_expressions:
            all_expr = np.vstack(list(self.niche_expressions.values()))
            self.expr_mean = np.mean(all_expr, axis=0)
            self.expr_std = np.std(all_expr, axis=0)
            self.expr_std[self.expr_std == 0] = 1.0
        
        if self.verbose:
            print(f"Computed expressions for {len(self.niche_expressions)} niches")
    
    def _crop_region_from_wsi(self, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        if self.wsi_image is None:
            return None
        
        x_min, y_min, x_max, y_max = bbox
        
        px_xmin = int(x_min * self.pixel_per_um)
        px_ymin = int(y_min * self.pixel_per_um)
        px_xmax = int(x_max * self.pixel_per_um)
        px_ymax = int(y_max * self.pixel_per_um)
        
        h, w = self.wsi_image.shape[:2]
        px_xmin = max(0, px_xmin)
        px_ymin = max(0, px_ymin)
        px_xmax = min(w, px_xmax)
        px_ymax = min(h, px_ymax)
        
        if px_xmax <= px_xmin or px_ymax <= px_ymin:
            return None
        
        crop = self.wsi_image[px_ymin:px_ymax, px_xmin:px_xmax]
        return crop
    
    def _load_region_from_lmdb(self, niche_id: str, row: Optional[pd.Series] = None) -> Optional[np.ndarray]:
        if self.lmdb_env is None:
            return None
        
        try:
            with self.lmdb_env.begin() as txn:
                key_patterns = []
                
                if row is not None:
                    grid_row = row.get('source_grid_row') or row.get('grid_row')
                    grid_col = row.get('source_grid_col') or row.get('grid_col')
                    if grid_row is not None and grid_col is not None:
                        dataset_name = getattr(self, 'dataset_name', 'Xenium_FFPE_Human_Breast_Cancer_Rep1_outs')
                        key_patterns.append(f"{dataset_name}/grid_{int(grid_row)}_{int(grid_col)}_256_256")
                
                if 'grid_' in str(niche_id):
                    dataset_name = getattr(self, 'dataset_name', 'Xenium_FFPE_Human_Breast_Cancer_Rep1_outs')
                    key_patterns.append(f"{dataset_name}/{niche_id}")
                
                key_patterns.extend([
                    f"niche_{niche_id}",
                    str(niche_id),
                    f"region_{niche_id}",
                ])
                
                for key_pattern in key_patterns:
                    data = txn.get(key_pattern.encode())
                    if data:
                        img = Image.open(io.BytesIO(data))
                        return np.array(img)
        except Exception as e:
            warnings.warn(f"Error loading niche {niche_id} from LMDB: {e}")
        
        return None
    
    def _get_region_image(self, row: pd.Series, prefix: str) -> Optional[torch.Tensor]:
        niche_id = row.get(f'{prefix}_niche_id', row.get(f'{prefix}_grid_id', ''))
        
        prefix_row = pd.Series({
            'grid_row': row.get(f'{prefix}_grid_row'),
            'grid_col': row.get(f'{prefix}_grid_col'),
        })
        
        img = self._load_region_from_lmdb(niche_id, prefix_row)
        
        if img is None and f'{prefix}_bbox_xmin' in row:
            bbox = (
                row[f'{prefix}_bbox_xmin'],
                row[f'{prefix}_bbox_ymin'],
                row[f'{prefix}_bbox_xmax'],
                row[f'{prefix}_bbox_ymax'],
            )
            if not any(pd.isna(bbox)):
                img = self._crop_region_from_wsi(bbox)
        
        if img is None:
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        if self.transforms:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = self.transforms(img)
        else:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        return img
    
    def _get_niche_expression(self, niche_id: str) -> torch.Tensor:
        if niche_id in self.niche_expressions:
            expr = self.niche_expressions[niche_id].copy()
            if self.normalize_expr:
                expr = (expr - self.expr_mean) / self.expr_std
                expr = np.clip(expr, -10, 10)
            return torch.from_numpy(expr)
        else:
            return torch.zeros(self.adata.n_vars if self.adata else 1000, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        row = self.pairs.iloc[idx]
        
        source_img = self._get_region_image(row, 'source')
        target_img = self._get_region_image(row, 'target')
        
        output = {
            'source_image': source_img,
            'target_image': target_img,
            'source_niche_id': str(row['source_niche_id']),
            'target_niche_id': str(row['target_niche_id']),
            'source_n_cells': int(row['source_n_cells']),
            'target_n_cells': int(row['target_n_cells']),
            'state_A': row['state_A'],
            'state_B': row['state_B'],
            'transition_tag': row['transition_tag'],
            'task_name': row.get('task_name', 'unknown'),
        }
        
        if self.return_expr:
            output['source_expr'] = self._get_niche_expression(row['source_niche_id'])
            output['target_expr'] = self._get_niche_expression(row['target_niche_id'])
        
        transition_tag = row['transition_tag']
        if transition_tag in self.delta_g_map:
            delta_g = self.delta_g_map[transition_tag].copy()
            if self.normalize_expr and hasattr(self, 'expr_std'):
                delta_g = delta_g / self.expr_std
                delta_g = np.clip(delta_g, -10, 10)
            output['delta_g'] = torch.from_numpy(delta_g)
        
        if 'source_bbox_xmin' in row and not pd.isna(row['source_bbox_xmin']):
            output['source_bbox'] = torch.tensor([
                row['source_bbox_xmin'], row['source_bbox_ymin'],
                row['source_bbox_xmax'], row['source_bbox_ymax']
            ], dtype=torch.float32)
        if 'target_bbox_xmin' in row and not pd.isna(row['target_bbox_xmin']):
            output['target_bbox'] = torch.tensor([
                row['target_bbox_xmin'], row['target_bbox_ymin'],
                row['target_bbox_xmax'], row['target_bbox_ymax']
            ], dtype=torch.float32)
        
        return output
    
    def get_stats(self) -> Dict:
        return {
            'n_pairs': len(self),
            'transitions': self.pairs['transition_tag'].value_counts().to_dict(),
            'avg_source_cells': self.pairs['source_n_cells'].mean(),
            'avg_target_cells': self.pairs['target_n_cells'].mean(),
            'has_delta_g': len(self.delta_g_map) > 0,
            'has_images': self.lmdb_env is not None or self.wsi_image is not None,
        }


def create_niche_dataloaders(
    dataset: SpatiaNicheDataset,
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    
    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )
        return train_loader, None


if __name__ == "__main__":
    print("SpatiaNicheDataset example usage:")
    print("""
    from spatia_niche_dataset import SpatiaNicheDataset, create_niche_dataloaders
    
    dataset = SpatiaNicheDataset(
        adata_path="/path/to/adata.h5ad",
        pairs_csv="niche_pairs_output/niche_pairs.csv",
        delta_g_npz="niche_pairs_output/niche_delta_g.npz",
        wsi_image_path="/path/to/he_image.tif",  # or use lmdb_path
        image_size=(512, 512),
        return_expr=True,
    )
    
    train_loader, val_loader = create_niche_dataloaders(dataset, batch_size=8)
    
    for batch in train_loader:
        source_imgs = batch['source_image']      # [B, C, H, W]
        target_imgs = batch['target_image']      # [B, C, H, W]
        source_expr = batch['source_expr']       # [B, G]
        delta_g = batch.get('delta_g')           # [B, G]
        # ... training code
    """)
