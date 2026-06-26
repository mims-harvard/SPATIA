# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent / "data_pairing_for_FM"))
from spatia_dataset import SpatiaPairedDataset
from .spatia_embedder import SpatiaEmbedder


class SpatiaBioDataLoader(LightningDataModule):
    
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        
        self.adata_path = getattr(args, 'adata_path', None)
        self.pairs_csv = getattr(args, 'pairs_csv', None)
        self.lmdb_path = getattr(args, 'lmdb_path', None)
        self.delta_g_npz = getattr(args, 'delta_g_npz', None)
        self.delta_m_npz = getattr(args, 'delta_m_npz', None)
        
        self._validate_paths()
        
        print("=" * 60)
        print("Initializing SPATIA Embedder...")
        print(f"  Model path: {args.spatia_model_path}")
        print(f"  Vocab path: {args.spatia_vocab_path}")
        print("  (This may take 1-2 minutes to load the model...)")
        print("=" * 60)
        import sys
        sys.stdout.flush()
        
        self.spatia_embedder = SpatiaEmbedder(
            model_path=args.spatia_model_path,
            vocab_path=args.spatia_vocab_path,
            gene_stats_path=args.spatia_gene_stats,
            device=device,
            batch_size=32,
            config={
                'dataset_name': getattr(args, 'dataset_name', 'xenium_data'),
                'lmdb_path': getattr(args, 'lmdb_path', None),
            }
        )
        self.spatia_embedder.args = args
        self.spatia_embedder.lmdb_path = getattr(args, 'lmdb_path', None)
        print("SPATIA Embedder initialized successfully!")
        sys.stdout.flush()
        
        print(f"\nLoading biological paired dataset from {self.pairs_csv}...")
        sys.stdout.flush()
        self.paired_dataset = SpatiaPairedDataset(
            adata_path=self.adata_path,
            pairs_csv=self.pairs_csv,
            lmdb_path=self.lmdb_path,
            delta_g_npz=self.delta_g_npz,
            delta_m_npz=self.delta_m_npz,
            return_expr=True,
            image_size=(256, 256)
        )
        
        print(f"Loaded {len(self.paired_dataset)} biological pairs")
        stats = self.paired_dataset.get_data_stats()
        if stats.get('has_delta_m'):
            print(f"  Δm signatures loaded: {stats.get('n_delta_m')} transitions")
        if 'ot_confidence_mean' in stats:
            print(f"  OT confidence: mean={stats['ot_confidence_mean']:.4f}, std={stats['ot_confidence_std']:.4f}")
        
        self.max_pairs = getattr(args, 'max_pairs', None)
        
        self.latent_dim = 512
        
        max_cells = len(self.paired_dataset.adata)
        self.embedding_matrix = torch.nn.Embedding(max_cells, self.latent_dim).to(device)
        
        self._create_mol_mapping()
        
        self._setup_splits()
    
    def _validate_paths(self):
        required = {
            'adata_path': self.adata_path,
            'pairs_csv': self.pairs_csv,
            'lmdb_path': self.lmdb_path,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"Missing required path arguments: {missing}. "
                           f"Please set them in your YAML config file.")
        
        for name, path in required.items():
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"{name} does not exist: {path}")
    
    def _create_mol_mapping(self):
        transitions = set()
        if 'transition_tag' in self.paired_dataset.pairs.columns:
            transitions = set(self.paired_dataset.pairs['transition_tag'].unique())
        
        self.mol2id = {t: i for i, t in enumerate(sorted(transitions))}
        self.id2mol = {v: k for k, v in self.mol2id.items()}
        
        self.num_transitions = len(self.mol2id)
        print(f"  Transition ID mapping created: {self.num_transitions} transitions")
    
    def _setup_splits(self):
        n_pairs = len(self.paired_dataset)
        
        if self.max_pairs and n_pairs > self.max_pairs:
            n_pairs = self.max_pairs
            print(f"Limiting to {self.max_pairs} pairs for testing")
            
        indices = np.random.permutation(n_pairs)
        
        split_idx = int(0.8 * n_pairs)
        
        self.train_indices = indices[:split_idx]
        self.test_indices = indices[split_idx:]
        
        print(f"Train pairs: {len(self.train_indices)}, Test pairs: {len(self.test_indices)}")
    
    def _prepare_datasets(self):
        if hasattr(self, '_datasets_prepared') and self._datasets_prepared:
            return
        
        print("Pre-generating SPATIA embeddings for all datasets...")
        
        self._train_dataset = SpatiaBioDataset(
            paired_dataset=self.paired_dataset,
            indices=self.train_indices,
            spatia_embedder=self.spatia_embedder,
            mol2id=self.mol2id,
            is_train=True
        )
        
        self._test_dataset = SpatiaBioDataset(
            paired_dataset=self.paired_dataset,
            indices=self.test_indices,
            spatia_embedder=self.spatia_embedder,
            mol2id=self.mol2id,
            is_train=False
        )
        
        if hasattr(self, 'spatia_embedder') and self.spatia_embedder is not None:
            print("Clearing SPATIA embedder from datamodule to free GPU memory...")
            self.spatia_embedder.clear_gpu_memory()
            del self.spatia_embedder
            self.spatia_embedder = None
            
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU memory cleared successfully")
        
        self._datasets_prepared = True
    
    def train_dataloader(self):
        self._prepare_datasets()
        
        return DataLoader(
            self._train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
    
    def test_dataloader(self):
        self._prepare_datasets()
        
        return DataLoader(
            self._test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )


class SpatiaBioDataset(Dataset):
    
    def __init__(
        self,
        paired_dataset: SpatiaPairedDataset,
        indices: np.ndarray,
        spatia_embedder: SpatiaEmbedder,
        mol2id: Dict[str, int],
        is_train: bool = True,
        return_delta_g: bool = True,
        return_delta_m: bool = True,
    ):
        self.paired_dataset = paired_dataset
        self.indices = indices
        self.spatia_embedder = spatia_embedder
        self.mol2id = mol2id
        self.is_train = is_train
        self.return_delta_g = return_delta_g
        self.return_delta_m = return_delta_m

        self._delta_g_projector = None
        if return_delta_g and paired_dataset.delta_g_map:
            self._fit_delta_g_pca(target_dim=512)

        print("Generating SPATIA embeddings for control cells...")
        self._generate_spatia_embeddings()
    
    def _fit_delta_g_pca(self, target_dim: int = 512):
        from sklearn.decomposition import PCA

        delta_g_map = self.paired_dataset.delta_g_map
        keys = list(delta_g_map.keys())
        raw_dim = len(np.asarray(delta_g_map[keys[0]]).flatten())

        if raw_dim <= target_dim:
            print(f"delta_g dim ({raw_dim}) <= target ({target_dim}), no PCA needed")
            self._delta_g_projector = None
            self._delta_g_target_dim = raw_dim
            return

        all_delta_g = []
        for idx in self.indices:
            sample = self.paired_dataset[idx]
            if 'delta_g' in sample:
                vec = sample['delta_g']
                if isinstance(vec, torch.Tensor):
                    vec = vec.numpy()
                all_delta_g.append(np.asarray(vec).flatten())

        if not all_delta_g:
            print("Warning: no delta_g samples found for PCA, skipping")
            self._delta_g_projector = None
            self._delta_g_target_dim = raw_dim
            return

        all_delta_g = np.stack(all_delta_g)
        n_components = min(target_dim, all_delta_g.shape[0], all_delta_g.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(all_delta_g)
        self._delta_g_projector = pca
        self._delta_g_target_dim = n_components
        explained = pca.explained_variance_ratio_.sum()
        print(f"delta_g PCA: {raw_dim} -> {n_components} dims "
              f"(explained variance: {explained:.3f})")

    def _project_delta_g(self, delta_g: torch.Tensor, target_dim: int = 512) -> torch.Tensor:
        if self._delta_g_projector is None:
            return delta_g
        vec = delta_g.numpy().reshape(1, -1)
        projected = self._delta_g_projector.transform(vec).flatten().astype(np.float32)
        if projected.shape[0] < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:projected.shape[0]] = projected
            projected = padded
        return torch.from_numpy(projected)

    def _generate_spatia_embeddings(self):
        self.spatia_embeddings = {}
        
        ctrl_indices = []
        for idx in self.indices:
            sample = self.paired_dataset[idx]
            ctrl_idx = sample['ctrl_idx']
            ctrl_indices.append(ctrl_idx)
        
        unique_ctrl_indices = list(set(ctrl_indices))
        print(f"Generating embeddings for {len(unique_ctrl_indices)} unique control cells")
        
        adata = self.spatia_embedder.preprocess_adata(self.paired_dataset.adata)
        
        original_obs_names = self.paired_dataset.adata.obs_names
        preprocessed_obs_names = adata.obs_names
        
        idx_mapping = []
        filtered_count = 0
        for ctrl_idx in unique_ctrl_indices:
            if ctrl_idx < len(original_obs_names):
                original_cell_name = original_obs_names[ctrl_idx]
                if original_cell_name in preprocessed_obs_names:
                    new_idx = preprocessed_obs_names.get_loc(original_cell_name)
                    idx_mapping.append((ctrl_idx, new_idx))
                else:
                    filtered_count += 1
            else:
                filtered_count += 1

        if filtered_count > 0:
            print(f"Note: {filtered_count} control cells were filtered during preprocessing")
        print(f"Generating embeddings for {len(idx_mapping)} control cells")

        if len(idx_mapping) == 0:
            print("Warning: No valid control cells found, using zero embeddings")
            for idx in unique_ctrl_indices:
                self.spatia_embeddings[idx] = np.zeros(512, dtype=np.float32)
            return

        print("Generating SPATIA embeddings in batch...")
        preprocessed_indices = [m[1] for m in idx_mapping]
        ctrl_adata = adata[preprocessed_indices].copy()
        batch_embeddings = self.spatia_embedder.generate_embeddings(
            ctrl_adata, return_numpy=True
        )

        n_processed = min(len(idx_mapping), batch_embeddings.shape[0])
        for i in range(n_processed):
            original_idx = idx_mapping[i][0]
            self.spatia_embeddings[original_idx] = batch_embeddings[i]
        
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        sample = self.paired_dataset[sample_idx]
        
        ctrl_image = sample['ctrl_image']
        tgt_image = sample['tgt_image']
        
        ctrl_idx = sample['ctrl_idx']
        if ctrl_idx in self.spatia_embeddings:
            spatia_embedding = torch.tensor(
                self.spatia_embeddings[ctrl_idx], 
                dtype=torch.float32
            )
        else:
            spatia_embedding = torch.zeros(512, dtype=torch.float32)
        
        transition_tag = sample.get('transition_tag', 'unknown')
        mol_id = self.mol2id.get(transition_tag, 0)
        
        output = {
            'X': (ctrl_image, tgt_image),
            'mols': torch.tensor([mol_id], dtype=torch.long),
            'y_id': torch.tensor([mol_id], dtype=torch.long),
            'concat_conditioning': spatia_embedding,
            'transition_ids': torch.tensor(mol_id, dtype=torch.long),
            'file_names': (f"ctrl_{ctrl_idx}", f"tgt_{sample['tgt_idx']}"),
        }
        
        output['ot_confidence'] = torch.tensor(
            sample.get('ot_confidence', 1.0), 
            dtype=torch.float32
        )
        
        if self.return_delta_g and 'delta_g' in sample:
            output['delta_g'] = self._project_delta_g(sample['delta_g'])
        else:
            dim_g = getattr(self, '_delta_g_target_dim', 512)
            output['delta_g'] = torch.zeros(dim_g, dtype=torch.float32)
        
        if self.return_delta_m and 'delta_m' in sample:
            output['delta_m'] = sample['delta_m']
        else:
            output['delta_m'] = torch.zeros(10, dtype=torch.float32)
        
        return output


def create_spatia_bio_dataloader(args, device):
    required_spatia_args = ['spatia_model_path', 'spatia_vocab_path', 'spatia_gene_stats']
    for arg in required_spatia_args:
        if not hasattr(args, arg) or getattr(args, arg) is None:
            raise ValueError(f"Required SPATIA argument '{arg}' is missing")
    
    return SpatiaBioDataLoader(args, device)
