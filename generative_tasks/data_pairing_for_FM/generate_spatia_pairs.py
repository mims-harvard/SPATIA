
import argparse
import io
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import cv2
import lmdb
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from skimage import measure, filters
from PIL import Image
import warnings


@dataclass
class PairingConfig:
    state_col: str = "cell_states"
    niche_col: str = "niche"
    expr_layer: Optional[str] = None
    pca_dim: int = 50
    sinkhorn_eps: float = 0.05
    sinkhorn_iter: int = 200
    min_cells_per_state: int = 5
    max_cells_per_group: int = 500
    seed: int = 42
    lmdb_path: Optional[str] = None


def extract_simple_morphology(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    try:
        thresh_val = filters.threshold_otsu(gray)
        mask = gray > thresh_val
    except ValueError:
        return np.zeros(10, dtype=np.float32)
    
    labels = measure.label(mask)
    if labels.max() == 0:
        return np.zeros(10, dtype=np.float32)
    
    regions = measure.regionprops(labels, intensity_image=gray)
    if not regions:
        return np.zeros(10, dtype=np.float32)
    
    main_region = max(regions, key=lambda r: r.area)
    
    try:
        features = np.array([
            main_region.area,
            main_region.perimeter if main_region.perimeter else 0.0,
            main_region.eccentricity if hasattr(main_region, 'eccentricity') else 0.0,
            main_region.solidity if hasattr(main_region, 'solidity') else 0.0,
            main_region.mean_intensity,
            main_region.max_intensity,
            main_region.min_intensity,
            np.std(gray[mask]) if np.any(mask) else 0.0,
            main_region.major_axis_length if hasattr(main_region, 'major_axis_length') else 0.0,
            main_region.minor_axis_length if hasattr(main_region, 'minor_axis_length') else 0.0,
        ], dtype=np.float32)
    except Exception:
        return np.zeros(10, dtype=np.float32)
    
    return features


def _normalize_dataset_name(name: str) -> str:
    base = str(name).split("/")[-1]
    if base.endswith(".h5ad"):
        base = base[:-5]
    return base


def read_lmdb_image(env: lmdb.Environment, key: str) -> np.ndarray:
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
            return np.zeros((128, 128, 3), dtype=np.uint8)
        
        meta = {}
        if meta_bytes:
            try:
                meta = json.loads(meta_bytes.decode("utf-8"))
            except:
                pass
        
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
                    side = int(np.sqrt(data.size // 3)) if data.size > 0 else 128
                    if data.size == side * side * 3:
                        arr = data.reshape(side, side, 3)
                    elif data.size == side * side:
                        arr = np.stack((data.reshape(side, side),) * 3, axis=-1)
                    else:
                        return np.zeros((128, 128, 3), dtype=np.uint8)
                if arr.ndim == 2:
                    arr = np.stack((arr,) * 3, axis=-1)
        except Exception:
            return np.zeros((128, 128, 3), dtype=np.uint8)
        
        return arr.astype(np.uint8)


BIOLOGICAL_TRANSITIONS = [
    ("Epi_FOXA1+", "EMT-Epi1_CEACAM6+", "EMT_transition", "tumor_progression"),
    ("Epi_FOXA1+", "Epi_CENPF+", "proliferation_activation", "tumor_progression"),
    ("Epi_FOXA1+", "mgEpi_KRT14+", "lineage_conversion", "tumor_progression"),
    ("tcm_CD4+T", "eff_CD8+T1", "T_cell_activation", "immune_infiltration"),
    ("EC_CAVIN2+", "EC_CLEC14A+", "angiogenesis_activation", "immune_infiltration"),
]


def get_expression_matrix(adata: sc.AnnData, indices: np.ndarray, 
                          expr_layer: Optional[str] = None) -> np.ndarray:
    X = adata.layers[expr_layer][indices] if expr_layer else adata.X[indices]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32)


def pca_project(X: np.ndarray, n_components: int) -> np.ndarray:
    if X.shape[1] <= n_components:
        return X
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return (U[:, :n_components] * S[:n_components]).astype(np.float32)


def sinkhorn_pairing(X_ctrl: np.ndarray, X_tgt: np.ndarray, 
                     eps: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    Xc = torch.from_numpy(X_ctrl)
    Xt = torch.from_numpy(X_tgt)
    
    with torch.no_grad():
        C = torch.cdist(Xc, Xt, p=2)
        
        Nc, Nt = C.shape
        mu = torch.full((Nc,), 1.0 / Nc, dtype=C.dtype)
        nu = torch.full((Nt,), 1.0 / Nt, dtype=C.dtype)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        for _ in range(n_iter):
            u = eps * (torch.log(mu) - torch.logsumexp((-C + u[:, None] + v[None, :]) / eps, dim=1)) + u
            v = eps * (torch.log(nu) - torch.logsumexp((-C + u[:, None] + v[None, :]) / eps, dim=0)) + v
        
        K = torch.exp((-C + u[:, None] + v[None, :]) / eps)
        
        max_vals, matched_indices = torch.max(K, dim=1)
    
    return matched_indices.cpu().numpy(), max_vals.cpu().numpy()


def generate_pairs_for_transition(
    adata: sc.AnnData,
    state_A: str,
    state_B: str,
    transition_tag: str,
    cfg: PairingConfig,
    rng: np.random.Generator,
) -> Tuple[List[dict], List[np.ndarray], List[np.ndarray]]:
    pair_records = []
    delta_g_vectors = []
    delta_m_vectors = []
    
    mask_A = adata.obs[cfg.state_col] == state_A
    mask_B = adata.obs[cfg.state_col] == state_B
    
    n_A = mask_A.sum()
    n_B = mask_B.sum()
    
    if n_A < cfg.min_cells_per_state or n_B < cfg.min_cells_per_state:
        print(f"    Skipping: insufficient cells ({state_A}: {n_A}, {state_B}: {n_B})")
        return pair_records, delta_g_vectors, delta_m_vectors
    
    niches_A = set(adata.obs.loc[mask_A, cfg.niche_col].unique())
    niches_B = set(adata.obs.loc[mask_B, cfg.niche_col].unique())
    shared_niches = niches_A & niches_B
    
    if not shared_niches:
        print(f"    No shared niches, using global pairing")
        shared_niches = {None}
    
    lmdb_env = None
    if cfg.lmdb_path and Path(cfg.lmdb_path).exists():
        try:
            lmdb_env = lmdb.open(cfg.lmdb_path, readonly=True, lock=False, subdir=False)
        except Exception as e:
            warnings.warn(f"Could not open LMDB at {cfg.lmdb_path}: {e}")
    
    for niche in shared_niches:
        if niche is not None:
            niche_mask = adata.obs[cfg.niche_col] == niche
            idx_ctrl = np.where((mask_A & niche_mask).values)[0]
            idx_tgt = np.where((mask_B & niche_mask).values)[0]
        else:
            idx_ctrl = np.where(mask_A.values)[0]
            idx_tgt = np.where(mask_B.values)[0]
        
        if len(idx_ctrl) < cfg.min_cells_per_state or len(idx_tgt) < cfg.min_cells_per_state:
            continue
        
        if len(idx_ctrl) > cfg.max_cells_per_group:
            idx_ctrl = rng.choice(idx_ctrl, cfg.max_cells_per_group, replace=False)
        if len(idx_tgt) > cfg.max_cells_per_group:
            idx_tgt = rng.choice(idx_tgt, cfg.max_cells_per_group, replace=False)
        
        X_ctrl = get_expression_matrix(adata, idx_ctrl, cfg.expr_layer)
        X_tgt = get_expression_matrix(adata, idx_tgt, cfg.expr_layer)
        
        X_combined = np.vstack([X_ctrl, X_tgt])
        X_pca = pca_project(X_combined, cfg.pca_dim)
        X_ctrl_pca = X_pca[:len(idx_ctrl)]
        X_tgt_pca = X_pca[len(idx_ctrl):]
        
        matched_indices, confidences = sinkhorn_pairing(
            X_ctrl_pca, X_tgt_pca, cfg.sinkhorn_eps, cfg.sinkhorn_iter
        )
        
        for i, j in enumerate(matched_indices):
            global_ctrl_idx = idx_ctrl[i]
            global_tgt_idx = idx_tgt[j]
            confidence = float(confidences[i])
            
            delta_m = np.zeros(10, dtype=np.float32)
            if lmdb_env is not None:
                try:
                    if "dataset_name" in adata.obs.columns:
                        ds_name = _normalize_dataset_name(str(adata.obs.iloc[global_ctrl_idx]["dataset_name"]))
                    else:
                        ds_name = "xenium_data"
                    
                    if "cell_id" in adata.obs.columns:
                        cell_id_ctrl = str(adata.obs.iloc[global_ctrl_idx]["cell_id"])
                        cell_id_tgt = str(adata.obs.iloc[global_tgt_idx]["cell_id"])
                    else:
                        cell_id_ctrl = str(global_ctrl_idx)
                        cell_id_tgt = str(global_tgt_idx)
                    
                    key_ctrl = f"{ds_name}/{cell_id_ctrl}"
                    key_tgt = f"{ds_name}/{cell_id_tgt}"
                    
                    img_ctrl = read_lmdb_image(lmdb_env, key_ctrl)
                    img_tgt = read_lmdb_image(lmdb_env, key_tgt)
                    
                    m_ctrl = extract_simple_morphology(img_ctrl)
                    m_tgt = extract_simple_morphology(img_tgt)
                    
                    delta_m = m_tgt - m_ctrl
                except Exception as e:
                    warnings.warn(f"Error extracting morphology for pair {i}: {e}")
            
            delta_m_vectors.append(delta_m)
            
            pair_records.append({
                "x_ctrl_id": int(global_ctrl_idx),
                "x_tgt_id": int(global_tgt_idx),
                "state_A": state_A,
                "state_B": state_B,
                "niche_ctrl": str(adata.obs.iloc[global_ctrl_idx][cfg.niche_col]),
                "niche_tgt": str(adata.obs.iloc[global_tgt_idx][cfg.niche_col]),
                "transition_tag": transition_tag,
                "ot_confidence": confidence,
            })
            
            delta = X_tgt[j] - X_ctrl[i]
            delta_g_vectors.append(delta)
    
    if lmdb_env is not None:
        lmdb_env.close()
    
    return pair_records, delta_g_vectors, delta_m_vectors


def run_pairing_pipeline(adata: sc.AnnData, cfg: PairingConfig) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    
    for col in [cfg.state_col, cfg.niche_col]:
        if col not in adata.obs.columns:
            raise ValueError(f"Required column '{col}' not found in adata.obs")
    
    all_pairs = []
    transition_delta_g: Dict[str, List[np.ndarray]] = {}
    transition_delta_m: Dict[str, List[np.ndarray]] = {}
    
    available_states = set(adata.obs[cfg.state_col].unique())
    
    print(f"\nDataset: {adata.shape[0]} cells, {adata.shape[1]} genes")
    print(f"Available states: {len(available_states)}")
    print(f"Niches: {len(adata.obs[cfg.niche_col].unique())}")
    if cfg.lmdb_path:
        print(f"LMDB path: {cfg.lmdb_path}")
    
    for state_A, state_B, tag, task in BIOLOGICAL_TRANSITIONS:
        print(f"\nProcessing {tag} ({state_A} → {state_B}):")
        
        if state_A not in available_states:
            print(f"    State '{state_A}' not found in data")
            continue
        if state_B not in available_states:
            print(f"    State '{state_B}' not found in data")
            continue
        
        pairs, deltas_g, deltas_m = generate_pairs_for_transition(
            adata, state_A, state_B, tag, cfg, rng
        )
        
        if pairs:
            for p in pairs:
                p["task_name"] = task
            
            all_pairs.extend(pairs)
            
            if tag not in transition_delta_g:
                transition_delta_g[tag] = []
            transition_delta_g[tag].extend(deltas_g)
            
            if tag not in transition_delta_m:
                transition_delta_m[tag] = []
            transition_delta_m[tag].extend(deltas_m)
            
            print(f"    Generated {len(pairs)} pairs")
        else:
            print(f"    No pairs generated")
    
    if not all_pairs:
        print("\nNo pairs generated!")
        return pd.DataFrame(), {}, {}
    
    pairs_df = pd.DataFrame(all_pairs)
    
    delta_g_dict = {}
    for tag, deltas in transition_delta_g.items():
        if deltas:
            stacked = np.vstack(deltas)
            delta_g_dict[tag] = np.mean(stacked, axis=0).astype(np.float32)
            print(f"\nΔg for {tag}: averaged over {len(deltas)} pairs, shape {delta_g_dict[tag].shape}")
    
    delta_m_dict = {}
    for tag, deltas in transition_delta_m.items():
        if deltas:
            stacked = np.vstack(deltas)
            delta_m_dict[tag] = np.mean(stacked, axis=0).astype(np.float32)
            print(f"Δm for {tag}: averaged over {len(deltas)} pairs, shape {delta_m_dict[tag].shape}")
    
    return pairs_df, delta_g_dict, delta_m_dict


def save_outputs(
    pairs_df: pd.DataFrame,
    delta_g_dict: Dict[str, np.ndarray],
    delta_m_dict: Dict[str, np.ndarray],
    adata: sc.AnnData,
    out_dir: Path,
    cfg: PairingConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pairs_csv = out_dir / "perturbation_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"\nSaved pairs: {pairs_csv}")
    
    delta_g_npz = out_dir / "delta_g_signatures.npz"
    np.savez_compressed(
        delta_g_npz,
        delta_g_keys=np.array(list(delta_g_dict.keys()), dtype=object),
        delta_g_values=np.array(list(delta_g_dict.values()), dtype=object),
        genes=np.array(list(adata.var_names)) if hasattr(adata, "var_names") else None,
    )
    print(f"Saved Δg signatures: {delta_g_npz}")
    
    if delta_m_dict:
        delta_m_npz = out_dir / "delta_m_signatures.npz"
        np.savez_compressed(
            delta_m_npz,
            **delta_m_dict
        )
        print(f"Saved Δm signatures: {delta_m_npz}")
    
    config_data = {
        "task_name": "spatia_perturbation_pairing",
        "description": "OT-based perturbation pairs for SPATIA flow matching with morphology shifts",
        "n_pairs": len(pairs_df),
        "transitions": pairs_df["transition_tag"].value_counts().to_dict(),
        "tasks": pairs_df["task_name"].value_counts().to_dict() if "task_name" in pairs_df.columns else {},
        "has_ot_confidence": "ot_confidence" in pairs_df.columns,
        "has_morphology": len(delta_m_dict) > 0,
        "morphology_dim": 10 if delta_m_dict else 0,
        "parameters": {
            "pca_dim": cfg.pca_dim,
            "sinkhorn_eps": cfg.sinkhorn_eps,
            "sinkhorn_iter": cfg.sinkhorn_iter,
            "min_cells_per_state": cfg.min_cells_per_state,
            "max_cells_per_group": cfg.max_cells_per_group,
            "seed": cfg.seed,
            "lmdb_path": cfg.lmdb_path,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    
    config_json = out_dir / "pairing_config.json"
    with open(config_json, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config: {config_json}")
    
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Total pairs: {len(pairs_df)}")
    if "ot_confidence" in pairs_df.columns:
        conf_stats = pairs_df["ot_confidence"]
        print(f"OT confidence: mean={conf_stats.mean():.4f}, std={conf_stats.std():.4f}")
    print("\nPairs per transition:")
    for tag, count in pairs_df["transition_tag"].value_counts().items():
        print(f"  {tag}: {count}")
    print("\nΔg signatures:")
    for tag, arr in delta_g_dict.items():
        print(f"  {tag}: shape {arr.shape}, range [{arr.min():.4f}, {arr.max():.4f}]")
    if delta_m_dict:
        print("\nΔm signatures:")
        for tag, arr in delta_m_dict.items():
            print(f"  {tag}: shape {arr.shape}, range [{arr.min():.4f}, {arr.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="SPATIA: OT-based Perturbation Pairing with Morphology Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_spatia_pairs.py --adata data.h5ad --out_dir ./output

  # With morphology extraction from LMDB images
  python generate_spatia_pairs.py --adata data.h5ad --lmdb_path images.lmdb --out_dir ./output

  # With custom columns
  python generate_spatia_pairs.py --adata data.h5ad --state_col cell_states --niche_col spatial_cluster --out_dir ./output
        """
    )
    
    parser.add_argument("--adata", required=True, help="Path to AnnData .h5ad file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--lmdb_path", default=None, help="Path to LMDB with cell images for morphology extraction")
    parser.add_argument("--state_col", default="cell_states", help="Column for cell states (default: cell_states)")
    parser.add_argument("--niche_col", default="niche", help="Column for spatial niche (default: niche)")
    parser.add_argument("--expr_layer", default=None, help="Expression layer in AnnData (default: X)")
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA dimensions for OT (default: 50)")
    parser.add_argument("--sinkhorn_eps", type=float, default=0.05, help="Sinkhorn regularization (default: 0.05)")
    parser.add_argument("--sinkhorn_iter", type=int, default=200, help="Sinkhorn iterations (default: 200)")
    parser.add_argument("--min_cells", type=int, default=5, help="Min cells per state per niche (default: 5)")
    parser.add_argument("--max_cells", type=int, default=500, help="Max cells per group (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    cfg = PairingConfig(
        state_col=args.state_col,
        niche_col=args.niche_col,
        expr_layer=args.expr_layer,
        pca_dim=args.pca_dim,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iter=args.sinkhorn_iter,
        min_cells_per_state=args.min_cells,
        max_cells_per_group=args.max_cells,
        seed=args.seed,
        lmdb_path=args.lmdb_path,
    )
    
    print("SPATIA Perturbation Pairing Pipeline")
    print("=" * 50)
    print(f"Input: {args.adata}")
    print(f"Output: {args.out_dir}")
    print(f"State column: {cfg.state_col}")
    print(f"Niche column: {cfg.niche_col}")
    if cfg.lmdb_path:
        print(f"LMDB path: {cfg.lmdb_path} (morphology extraction enabled)")
    else:
        print("LMDB path: None (morphology extraction disabled)")
    
    print(f"\nLoading AnnData...")
    adata = sc.read_h5ad(args.adata)
    
    pairs_df, delta_g_dict, delta_m_dict = run_pairing_pipeline(adata, cfg)
    
    if len(pairs_df) == 0:
        print("\nNo pairs generated. Check your state/niche columns and transition definitions.")
        return
    
    save_outputs(pairs_df, delta_g_dict, delta_m_dict, adata, Path(args.out_dir), cfg)


if __name__ == "__main__":
    main()
