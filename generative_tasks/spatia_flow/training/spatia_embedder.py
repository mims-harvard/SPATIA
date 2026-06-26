# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import scanpy as sc
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "gene_encoders" / "SPATIA-scgpt"))
from scgpt_spatial.model.model import TransformerModel
from scgpt_spatial.tokenizer import tokenize_and_pad_batch


class SpatiaEmbedder:
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        gene_stats_path: str,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size: int = 32,
        config: dict = None,
    ):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.gene_stats_path = gene_stats_path
        self.device = device
        self.batch_size = batch_size
        
        self.config = config or {}
        self.args = None
        self.dataset_name = self.config.get('dataset_name', 'xenium_data')
        self.lmdb_path = self.config.get('lmdb_path', None)
        
        import sys
        
        print("  [1/3] Loading vocabulary...", end=" ", flush=True)
        self.vocab = self._load_vocab()
        print(f"done ({len(self.vocab)} tokens)")
        sys.stdout.flush()
        
        print("  [2/3] Loading gene statistics...", end=" ", flush=True)
        self.gene_stats = self._load_gene_stats()
        print(f"done ({len(self.gene_stats)} genes)")
        sys.stdout.flush()
        
        print("  [3/3] Loading SPATIA Transformer model...")
        print("        (This includes ViT image encoder from HuggingFace)")
        sys.stdout.flush()
        self.model = self._load_model()
        print("  Model loaded successfully!")
        sys.stdout.flush()
        
    def _load_vocab(self) -> Dict:
        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab
    
    def _load_gene_stats(self) -> pd.DataFrame:
        return pd.read_csv(self.gene_stats_path, index_col=0)
    
    def _load_model(self) -> TransformerModel:
        import sys
        
        model_config = {
            "ntoken": len(self.vocab),
            "d_model": 512,
            "nhead": 8,
            "d_hid": 512,
            "nlayers": 12,
            "nlayers_cls": 3,
            "n_cls": 10,
            "vocab": self.vocab,
            "dropout": 0.2,
            "pad_token": "<pad>",
            "pad_value": 0,
            "do_mvc": False,
            "do_dab": False,
            "use_batch_labels": False,
            "input_emb_style": "continuous",
            "cell_emb_style": "cls",
            "explicit_zero_prob": False,
            "use_generative_training": False,
            "use_fast_transformer": False,
            "fast_transformer_backend": 'linear',
            "pre_norm": False,
            "image_encoder_cls": "facebook/vit-mae-base",
            "combine_weight": 1.0,
            "image_combine_weight": 1.0,
            "image_recon_loss_weight": 1.0,
        }
        
        print("        Initializing TransformerModel architecture...", flush=True)
        model = TransformerModel(**model_config)
        print("        Architecture initialized.", flush=True)
        
        checkpoint_path = os.path.join(self.model_path, "best_model.pt")
        print(f"        Looking for checkpoint at: {checkpoint_path}", flush=True)
        
        if os.path.exists(checkpoint_path):
            print(f"        Loading checkpoint (this may take a moment)...", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"        Checkpoint loaded, applying to model...", flush=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            
            if missing_keys:
                print(f"        Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"        Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                
        else:
            ckpt_files = list(Path(self.model_path).glob("*.pt"))
            print(f"        best_model.pt not found, looking for other checkpoints...", flush=True)
            if ckpt_files:
                print(f"        Found: {ckpt_files[0]}", flush=True)
                checkpoint = torch.load(ckpt_files[0], map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                    
                if missing_keys:
                    print(f"        Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"        Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            else:
                raise FileNotFoundError(f"No checkpoint found in {self.model_path}")
        
        print(f"        Moving model to {self.device}...", flush=True)
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_adata(
        self, 
        adata: sc.AnnData,
        filter_gene_by_counts: int = 3,
        filter_cell_by_counts: int = 200,
        normalize_total: float = 1e4,
        log1p: bool = True,
    ) -> sc.AnnData:
        adata = adata.copy()
        
        if 'symbol' in adata.var.columns:
            print("Converting Ensembl IDs to gene symbols...")
            adata.var['ensembl_id'] = adata.var.index
            
            valid_symbols = adata.var['symbol'].notna() & (adata.var['symbol'] != '') & (adata.var['symbol'] != 'nan')
            adata = adata[:, valid_symbols].copy()
            
            adata.var.index = adata.var['symbol'].astype(str)
            print(f"Converted {adata.n_vars} genes from Ensembl ID to gene symbols")
            print(f"Sample gene symbols: {list(adata.var.index[:10])}")
        else:
            print("Warning: No 'symbol' column found, keeping original gene names")
        
        sc.pp.filter_genes(adata, min_counts=filter_gene_by_counts)
        sc.pp.filter_cells(adata, min_counts=filter_cell_by_counts)
        
        adata.raw = adata
        
        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=normalize_total)
        
        if log1p:
            sc.pp.log1p(adata)
            
        return adata
    
    def tokenize_batch(
        self, 
        gene_names: list, 
        values: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gene_tokens = []
        for gene in gene_names:
            if gene in self.vocab:
                gene_tokens.append(self.vocab[gene])
            else:
                gene_tokens.append(self.vocab.get("<unk>", 0))
        
        data_array = values
        gene_ids_array = np.array(gene_tokens)
        
        tokenized = tokenize_and_pad_batch(
            data_array,
            gene_ids=gene_ids_array,
            max_len=len(gene_tokens),
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=0,
            append_cls=True,
            include_zero_gene=True,
        )
        
        seq_len = tokenized["genes"].shape[-1]
        batch_size = tokenized["genes"].shape[0]
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        
        return (
            torch.tensor(tokenized["genes"], dtype=torch.long),
            torch.tensor(tokenized["values"], dtype=torch.float),
            attention_mask
        )
    
    def _load_batch_images_from_lmdb(self, batch_indices, adata=None):
        try:
            import lmdb
            from transformers import AutoImageProcessor
            from pathlib import Path
            
            if not hasattr(self, 'lmdb_path') or self.lmdb_path is None:
                if hasattr(self, 'args') and hasattr(self.args, 'lmdb_path'):
                    self.lmdb_path = self.args.lmdb_path
                elif hasattr(self, 'args') and hasattr(self.args, 'spatial_datadir'):
                    self.lmdb_path = self.args.spatial_datadir
                elif hasattr(self, 'config') and 'lmdb_path' in self.config:
                    self.lmdb_path = self.config['lmdb_path']
                elif hasattr(self, 'config') and 'spatial_datadir' in self.config:
                    self.lmdb_path = self.config['spatial_datadir']
                else:
                    print("Warning: No LMDB path found, using dummy images")
                    return torch.zeros((len(batch_indices), 3, 224, 224), device=self.device)
            
            if not hasattr(self, 'image_preprocessor'):
                self.image_preprocessor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
            
            env = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=False)
            
            batch_images = []
            dataset_name = getattr(self, 'dataset_name', 'xenium_data')
            
            for idx in batch_indices:
                with env.begin() as txn:
                    if adata is not None:
                        if "cell_id" in adata.obs.columns:
                            cell_id = str(adata.obs.iloc[idx]["cell_id"])
                        elif "index" in adata.obs.columns:
                            cell_id = str(adata.obs.iloc[idx]["index"])
                        else:
                            cell_id = str(adata.obs.index[idx])
                        
                        if "dataset_name" in adata.obs.columns:
                            ds_name = str(adata.obs.iloc[idx]["dataset_name"]).split("/")[-1]
                            if ds_name.endswith(".h5ad"):
                                ds_name = ds_name[:-5]
                        else:
                            ds_name = dataset_name
                        
                        key = f"{ds_name}/{cell_id}"
                    else:
                        key = f"{dataset_name}/{idx}"
                    
                    value = None
                    for key_pattern in [key, f"{key}:image", f"{key}/image"]:
                        value = txn.get(key_pattern.encode("utf-8"))
                        if value is not None:
                            break
                    
                    if value is not None:
                        try:
                            from PIL import Image
                            import io
                            
                            try:
                                img = Image.open(io.BytesIO(value)).convert("RGB")
                                data = np.array(img)
                            except:
                                data = np.frombuffer(value, dtype=np.uint8)
                                if data.size == 256 * 256:
                                    data = data.reshape(256, 256)
                                    data = np.stack((data,) * 3, axis=-1)
                                elif data.size == 256 * 256 * 3:
                                    data = data.reshape(256, 256, 3)
                                else:
                                    side = int(np.sqrt(data.size // 3))
                                    data = data[:side*side*3].reshape(side, side, 3)
                            
                            image = self.image_preprocessor(
                                images=data, 
                                return_tensors="pt"
                            )["pixel_values"][0]
                            
                            batch_images.append(image)
                        except Exception as e:
                            print(f"Warning: Could not decode image for key {key}: {e}")
                            dummy_data = np.zeros((224, 224, 3), dtype=np.uint8)
                            image = self.image_preprocessor(images=dummy_data, return_tensors="pt")["pixel_values"][0]
                            batch_images.append(image)
                    else:
                        print(f"Warning: Image not found for key {key}, using dummy image")
                        dummy_data = np.zeros((224, 224, 3), dtype=np.uint8)
                        image = self.image_preprocessor(images=dummy_data, return_tensors="pt")["pixel_values"][0]
                        batch_images.append(image)
            
            env.close()
            
            if batch_images:
                stacked = torch.stack(batch_images).to(self.device)
                print(f"        Loaded {len(batch_images)} images, shape: {stacked.shape}")
                return stacked
            else:
                print("        Warning: No images loaded, using dummy images (224x224)")
                return torch.zeros((len(batch_indices), 3, 224, 224), device=self.device)
                
        except Exception as e:
            print(f"Error loading images from LMDB: {e}")
            import traceback
            traceback.print_exc()
            print("Using dummy images as fallback (224x224)")
            return torch.zeros((len(batch_indices), 3, 224, 224), device=self.device)
    
    def generate_embeddings(
        self, 
        adata: sc.AnnData,
        image_data: Optional[torch.Tensor] = None,
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        data_genes = adata.var_names.tolist()
        vocab_genes = [g for g in data_genes if g in self.vocab]
        
        if len(vocab_genes) == 0:
            raise ValueError("No common genes found between data and model vocabulary")
        
        adata_filtered = adata[:, vocab_genes].copy()
        
        if hasattr(adata_filtered, 'raw') and adata_filtered.raw is not None:
            X = adata_filtered.raw.X[:, adata_filtered.raw.var_names.isin(vocab_genes)]
        else:
            X = adata_filtered.X
            
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        all_embeddings = []
        n_cells = X.shape[0]
        
        with torch.no_grad():
            for start_idx in range(0, n_cells, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_cells)
                batch_X = X[start_idx:end_idx]
                
                gene_tokens, value_tokens, attention_mask = self.tokenize_batch(
                    vocab_genes, batch_X
                )
                
                gene_tokens = gene_tokens.to(self.device)
                value_tokens = value_tokens.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                if image_data is not None:
                    batch_images = image_data[start_idx:end_idx]
                else:
                    batch_cell_indices = list(range(start_idx, end_idx))
                    batch_images = self._load_batch_images_from_lmdb(batch_cell_indices, adata=adata_filtered)
                
                outputs = self.model.perceptual_forward(
                    src=gene_tokens,
                    values=value_tokens,
                    src_key_padding_mask=attention_mask,
                    image=batch_images,
                )
                
                batch_embeddings = outputs["cell_emb"]
                
                if return_numpy:
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                all_embeddings.append(batch_embeddings)
        
        if return_numpy:
            embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            embeddings = torch.cat(all_embeddings, dim=0)
            
        return embeddings
    
    def compute_differential_expression(
        self, 
        adata: sc.AnnData,
        state_col: str,
        state_A: str,
        state_B: str,
        cell_type_col: str = None,
        n_top_genes: int = 100,
    ) -> Dict[str, np.ndarray]:
        import scipy.stats as stats
        from sklearn.decomposition import PCA
        
        de_signatures = {}
        
        if cell_type_col and cell_type_col in adata.obs.columns:
            cell_types = adata.obs[cell_type_col].unique()
        else:
            cell_types = ['all']
            
        for cell_type in cell_types:
            if cell_type == 'all':
                adata_subset = adata
            else:
                adata_subset = adata[adata.obs[cell_type_col] == cell_type]
                
            cells_A = adata_subset[adata_subset.obs[state_col] == state_A]
            cells_B = adata_subset[adata_subset.obs[state_col] == state_B]
            
            if len(cells_A) < 10 or len(cells_B) < 10:
                print(f"Warning: Insufficient cells for {cell_type} ({len(cells_A)} vs {len(cells_B)})")
                continue
                
            X_A = cells_A.X.toarray() if hasattr(cells_A.X, 'toarray') else cells_A.X
            X_B = cells_B.X.toarray() if hasattr(cells_B.X, 'toarray') else cells_B.X
            
            pvals = []
            logfc = []
            
            for i in range(X_A.shape[1]):
                try:
                    stat, pval = stats.ranksums(X_A[:, i], X_B[:, i])
                    mean_A = np.mean(X_A[:, i])
                    mean_B = np.mean(X_B[:, i])
                    fc = np.log2((mean_B + 1e-8) / (mean_A + 1e-8))
                    
                    pvals.append(pval)
                    logfc.append(fc)
                except:
                    pvals.append(1.0)
                    logfc.append(0.0)
            
            pvals = np.array(pvals)
            logfc = np.array(logfc)
            
            significant = pvals < 0.05
            abs_logfc = np.abs(logfc)
            
            if np.sum(significant) > 0:
                top_indices = np.where(significant)[0]
                top_indices = top_indices[np.argsort(abs_logfc[significant])[::-1]]
            else:
                top_indices = np.argsort(abs_logfc)[::-1]
                
            top_indices = top_indices[:n_top_genes]
            
            de_vector = np.zeros(len(adata_subset.var_names))
            de_vector[top_indices] = logfc[top_indices]
            
            if len(top_indices) > 32:
                pca = PCA(n_components=32)
                de_reduced = pca.fit_transform(de_vector.reshape(1, -1)).flatten()
            else:
                de_reduced = de_vector[top_indices]
                if len(de_reduced) < 32:
                    de_reduced = np.pad(de_reduced, (0, 32 - len(de_reduced)))
                    
            de_signatures[cell_type] = de_reduced
            
        return de_signatures
    
    def clear_gpu_memory(self):
        import gc
        
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model = self.model.cpu()
            except:
                pass
            
            del self.model
            self.model = None
        
        if hasattr(self, 'image_preprocessor'):
            del self.image_preprocessor
            self.image_preprocessor = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("SPATIA model cleared from GPU memory")


