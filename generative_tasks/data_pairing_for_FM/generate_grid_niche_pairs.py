
import argparse
import io
import json
import numpy as np
import pandas as pd
import scanpy as sc
import time
import torch
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    import cv2
    import lmdb
    from skimage import measure, filters
    from PIL import Image
    MORPHOLOGY_AVAILABLE = True
except ImportError:
    MORPHOLOGY_AVAILABLE = False
    warnings.warn("cv2/lmdb/skimage not available - morphology extraction disabled")


@dataclass
class GridPairingConfig:
    state_col: str = "cell_states"
    cell_id_col: str = "index"
    
    grid_size: int = 256
    
    pca_dim: int = 50
    sinkhorn_eps: float = 0.05
    sinkhorn_iter: int = 200
    
    min_cells_per_grid: int = 5
    min_source_fraction: float = 0.05
    max_pairs_per_transition: int = 200
    
    lmdb_path: Optional[str] = None
    dataset_prefix: str = "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs"
    
    include_ot_confidence: bool = False
    
    seed: int = 42


BIOLOGICAL_TRANSITIONS = [
    ("Epi_FOXA1+", "EMT-Epi1_CEACAM6+", "EMT_transition", "tumor_progression"),
    ("Epi_FOXA1+", "Epi_CENPF+", "proliferation_activation", "tumor_progression"),
    ("Epi_FOXA1+", "mgEpi_KRT14+", "lineage_conversion", "tumor_progression"),
    ("tcm_CD4+T", "eff_CD8+T1", "T_cell_activation", "immune_infiltration"),
    ("EC_CAVIN2+", "EC_CLEC14A+", "angiogenesis_activation", "immune_infiltration"),
]


def pca_project(X: np.ndarray, n_components: int) -> np.ndarray:
    if X.shape[1] <= n_components:
        return X
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return (U[:, :n_components] * S[:n_components]).astype(np.float32)


def sinkhorn_pairing(X_source: np.ndarray, X_target: np.ndarray,
                     eps: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs = torch.from_numpy(X_source)
    Xt = torch.from_numpy(X_target)
    
    with torch.no_grad():
        C = torch.cdist(Xs, Xt, p=2)
        Ns, Nt = C.shape
        mu = torch.full((Ns,), 1.0 / Ns, dtype=C.dtype)
        nu = torch.full((Nt,), 1.0 / Nt, dtype=C.dtype)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        for _ in range(n_iter):
            u = eps * (torch.log(mu) - torch.logsumexp((-C + u[:, None] + v[None, :]) / eps, dim=1)) + u
            v = eps * (torch.log(nu) - torch.logsumexp((-C + u[:, None] + v[None, :]) / eps, dim=0)) + v
        
        K = torch.exp((-C + u[:, None] + v[None, :]) / eps)
        matched_indices = torch.argmax(K, dim=1).cpu().numpy()
        
        confidences = torch.max(K, dim=1).values.cpu().numpy()
    
    return matched_indices, confidences


def extract_grid_morphology(image_rgb: np.ndarray) -> np.ndarray:
    if not MORPHOLOGY_AVAILABLE:
        return np.zeros(10, dtype=np.float32)
    
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    
    if image_rgb.ndim == 2:
        gray = image_rgb
    else:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    try:
        thresh_val = filters.threshold_otsu(gray)
        mask = gray > thresh_val
    except ValueError:
        return np.zeros(10, dtype=np.float32)
    
    labels = measure.label(mask)
    if labels.max() == 0:
        return np.array([
            0, 0, 0, 0,
            np.mean(gray), np.max(gray), np.min(gray), np.std(gray),
            0, 0
        ], dtype=np.float32)
    
    regions = measure.regionprops(labels, intensity_image=gray)
    if not regions:
        return np.zeros(10, dtype=np.float32)
    
    total_area = sum(r.area for r in regions)
    total_perimeter = sum(r.perimeter if r.perimeter else 0 for r in regions)
    
    eccentricities = [r.eccentricity if hasattr(r, 'eccentricity') else 0 for r in regions]
    solidities = [r.solidity if hasattr(r, 'solidity') else 0 for r in regions]
    major_axes = [r.major_axis_length if hasattr(r, 'major_axis_length') else 0 for r in regions]
    minor_axes = [r.minor_axis_length if hasattr(r, 'minor_axis_length') else 0 for r in regions]
    
    areas = [r.area for r in regions]
    total = sum(areas)
    if total > 0:
        mean_ecc = sum(e * a for e, a in zip(eccentricities, areas)) / total
        mean_sol = sum(s * a for s, a in zip(solidities, areas)) / total
        mean_major = sum(m * a for m, a in zip(major_axes, areas)) / total
        mean_minor = sum(m * a for m, a in zip(minor_axes, areas)) / total
    else:
        mean_ecc = mean_sol = mean_major = mean_minor = 0
    
    try:
        features = np.array([
            total_area,
            total_perimeter,
            mean_ecc,
            mean_sol,
            np.mean(gray),
            np.max(gray),
            np.min(gray),
            np.std(gray),
            mean_major,
            mean_minor,
        ], dtype=np.float32)
    except Exception:
        return np.zeros(10, dtype=np.float32)
    
    return features


def read_grid_lmdb_image(env, key: str, grid_size: int = 256) -> np.ndarray:
    if not MORPHOLOGY_AVAILABLE:
        return np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
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
            return np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        meta = {}
        if meta_bytes:
            try:
                meta = json.loads(meta_bytes.decode("utf-8"))
            except:
                pass
        
        fmt = meta.get("format", "raw").lower()
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
                    side = int(np.sqrt(data.size // 3)) if data.size > 0 else grid_size
                    if data.size == side * side * 3:
                        arr = data.reshape(side, side, 3)
                    elif data.size == side * side:
                        arr = np.stack((data.reshape(side, side),) * 3, axis=-1)
                    else:
                        return np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
                
                if arr.ndim == 2:
                    arr = np.stack((arr,) * 3, axis=-1)
        except Exception as e:
            return np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        return arr.astype(np.uint8)


def compute_grid_state_composition(
    cell_adata: sc.AnnData,
    niche_adata: sc.AnnData,
    cfg: GridPairingConfig,
) -> Dict[str, Dict[str, float]]:
    print("Computing cell state composition per grid...")
    
    if cfg.cell_id_col in cell_adata.obs.columns:
        cell_id_to_state = dict(zip(
            cell_adata.obs[cfg.cell_id_col].astype(str).values,
            cell_adata.obs[cfg.state_col].values
        ))
        print(f"  Using '{cfg.cell_id_col}' column for cell ID mapping")
    else:
        cell_id_to_state = dict(zip(
            cell_adata.obs.index.astype(str),
            cell_adata.obs[cfg.state_col].values
        ))
        print(f"  Using index for cell ID mapping ('{cfg.cell_id_col}' not found)")
    
    print(f"  Built mapping for {len(cell_id_to_state)} cells")
    
    grid_state_probs: Dict[str, Dict[str, float]] = {}
    n_grids_processed = 0
    n_grids_with_cells = 0
    
    for idx, row in tqdm(niche_adata.obs.iterrows(), total=len(niche_adata.obs), 
                         desc="Processing grids"):
        grid_row = int(row['grid_row'])
        grid_col = int(row['grid_col'])
        grid_id = f"grid_{grid_row}_{grid_col}"
        
        cell_ids_str = row.get('cell_ids', '')
        if pd.isna(cell_ids_str) or not str(cell_ids_str).strip():
            continue
            
        cell_ids = [cid.strip() for cid in str(cell_ids_str).split(',') if cid.strip()]
        
        if len(cell_ids) < cfg.min_cells_per_grid:
            continue
        
        state_counts: Dict[str, int] = {}
        for cid in cell_ids:
            state = cell_id_to_state.get(cid)
            if state is not None and not pd.isna(state):
                if state not in state_counts:
                    state_counts[state] = 0
                state_counts[state] += 1
        
        total = sum(state_counts.values())
        if total >= cfg.min_cells_per_grid:
            grid_state_probs[grid_id] = {s: c / total for s, c in state_counts.items()}
            n_grids_with_cells += 1
        
        n_grids_processed += 1
    
    print(f"Processed {n_grids_processed} grids, {n_grids_with_cells} have sufficient cells")
    print(f"Computed state composition for {len(grid_state_probs)} grids")
    return grid_state_probs


def classify_grid_role(state_probs: Dict[str, float], state_A: str, state_B: str,
                       min_fraction: float) -> str:
    prob_A = state_probs.get(state_A, 0)
    prob_B = state_probs.get(state_B, 0)
    
    if prob_A >= min_fraction and prob_B < min_fraction:
        return 'source'
    elif prob_B >= min_fraction:
        return 'target'
    return 'other'


def generate_grid_pairs(
    niche_adata: sc.AnnData,
    grid_state_probs: Dict[str, Dict[str, float]],
    cfg: GridPairingConfig,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    
    rng = np.random.default_rng(cfg.seed)
    
    grid_id_to_idx = {}
    for i, row in niche_adata.obs.iterrows():
        grid_id = f"grid_{int(row['grid_row'])}_{int(row['grid_col'])}"
        grid_id_to_idx[grid_id] = niche_adata.obs.index.get_loc(i)
    
    X = niche_adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)
    
    lmdb_env = None
    extract_morphology = False
    if cfg.lmdb_path and MORPHOLOGY_AVAILABLE:
        try:
            lmdb_env = lmdb.open(cfg.lmdb_path, readonly=True, lock=False, subdir=False)
            extract_morphology = True
            print(f"Opened LMDB at {cfg.lmdb_path} for morphology extraction")
        except Exception as e:
            print(f"Warning: Could not open LMDB at {cfg.lmdb_path}: {e}")
            print("Continuing without morphology extraction")
    
    all_pairs = []
    transition_delta_g: Dict[str, List[np.ndarray]] = {}
    transition_delta_m: Dict[str, List[np.ndarray]] = {}
    
    available_states = set()
    for probs in grid_state_probs.values():
        available_states.update(probs.keys())
    
    for state_A, state_B, tag, task in BIOLOGICAL_TRANSITIONS:
        print(f"\nProcessing {tag} ({state_A} → {state_B}):")
        
        if state_A not in available_states or state_B not in available_states:
            print(f"  States not found in grid data")
            continue
        
        source_grids = []
        target_grids = []
        
        for grid_id, probs in grid_state_probs.items():
            if grid_id not in grid_id_to_idx:
                continue
            role = classify_grid_role(probs, state_A, state_B, cfg.min_source_fraction)
            if role == 'source':
                source_grids.append(grid_id)
            elif role == 'target':
                target_grids.append(grid_id)
        
        print(f"  Source grids: {len(source_grids)}, Target grids: {len(target_grids)}")
        
        if len(source_grids) < 2 or len(target_grids) < 2:
            print(f"  Skipping: insufficient grids")
            continue
        
        if len(source_grids) > cfg.max_pairs_per_transition:
            source_grids = list(rng.choice(source_grids, cfg.max_pairs_per_transition, replace=False))
        if len(target_grids) > cfg.max_pairs_per_transition:
            target_grids = list(rng.choice(target_grids, cfg.max_pairs_per_transition, replace=False))
        
        source_idx = [grid_id_to_idx[g] for g in source_grids]
        target_idx = [grid_id_to_idx[g] for g in target_grids]
        
        source_expr = X[source_idx]
        target_expr = X[target_idx]
        
        combined = np.vstack([source_expr, target_expr])
        combined_pca = pca_project(combined, cfg.pca_dim)
        source_pca = combined_pca[:len(source_grids)]
        target_pca = combined_pca[len(source_grids):]
        
        matched_indices, confidences = sinkhorn_pairing(
            source_pca, target_pca, cfg.sinkhorn_eps, cfg.sinkhorn_iter
        )
        
        source_morphology = {}
        target_morphology = {}
        if extract_morphology and lmdb_env is not None:
            print(f"  Extracting morphology features...")
            for grid_id in source_grids:
                row = int(grid_id.split('_')[1])
                col = int(grid_id.split('_')[2])
                lmdb_key = f"{cfg.dataset_prefix}/grid_{row}_{col}_{cfg.grid_size}_{cfg.grid_size}"
                img = read_grid_lmdb_image(lmdb_env, lmdb_key, cfg.grid_size)
                source_morphology[grid_id] = extract_grid_morphology(img)
            
            for grid_id in target_grids:
                row = int(grid_id.split('_')[1])
                col = int(grid_id.split('_')[2])
                lmdb_key = f"{cfg.dataset_prefix}/grid_{row}_{col}_{cfg.grid_size}_{cfg.grid_size}"
                img = read_grid_lmdb_image(lmdb_env, lmdb_key, cfg.grid_size)
                target_morphology[grid_id] = extract_grid_morphology(img)
        
        delta_m_vectors = []
        for i, j in enumerate(matched_indices):
            src_grid = source_grids[i]
            tgt_grid = target_grids[j]
            
            src_idx_local = grid_id_to_idx[src_grid]
            tgt_idx_local = grid_id_to_idx[tgt_grid]
            
            src_row = niche_adata.obs.iloc[src_idx_local]
            tgt_row = niche_adata.obs.iloc[tgt_idx_local]
            
            record = {
                'source_grid_id': src_grid,
                'target_grid_id': tgt_grid,
                'source_grid_row': int(src_row['grid_row']),
                'source_grid_col': int(src_row['grid_col']),
                'target_grid_row': int(tgt_row['grid_row']),
                'target_grid_col': int(tgt_row['grid_col']),
                'source_num_cells': int(src_row.get('num_cells', 0)),
                'target_num_cells': int(tgt_row.get('num_cells', 0)),
                'source_state_A_frac': grid_state_probs[src_grid].get(state_A, 0),
                'source_state_B_frac': grid_state_probs[src_grid].get(state_B, 0),
                'target_state_A_frac': grid_state_probs[tgt_grid].get(state_A, 0),
                'target_state_B_frac': grid_state_probs[tgt_grid].get(state_B, 0),
                'state_A': state_A,
                'state_B': state_B,
                'transition_tag': tag,
                'task_name': task,
            }
            if cfg.include_ot_confidence:
                record['ot_confidence'] = float(confidences[i])
            all_pairs.append(record)
            
            delta = target_expr[j] - source_expr[i]
            if tag not in transition_delta_g:
                transition_delta_g[tag] = []
            transition_delta_g[tag].append(delta)
            
            if extract_morphology and src_grid in source_morphology and tgt_grid in target_morphology:
                m_src = source_morphology[src_grid]
                m_tgt = target_morphology[tgt_grid]
                delta_m = m_tgt - m_src
                delta_m_vectors.append(delta_m)
        
        if delta_m_vectors:
            if tag not in transition_delta_m:
                transition_delta_m[tag] = []
            transition_delta_m[tag].extend(delta_m_vectors)
        
        print(f"  Generated {len(matched_indices)} pairs")
        if extract_morphology:
            print(f"  Extracted morphology for {len(delta_m_vectors)} pairs")
    
    if lmdb_env is not None:
        lmdb_env.close()
    
    if not all_pairs:
        return pd.DataFrame(), {}, {}
    
    pairs_df = pd.DataFrame(all_pairs)
    
    delta_g_dict = {}
    for tag, deltas in transition_delta_g.items():
        if deltas:
            delta_g_dict[tag] = np.mean(np.vstack(deltas), axis=0).astype(np.float32)
    
    delta_m_dict = {}
    for tag, deltas in transition_delta_m.items():
        if deltas:
            delta_m_dict[tag] = np.mean(np.vstack(deltas), axis=0).astype(np.float32)
    
    return pairs_df, delta_g_dict, delta_m_dict


def save_outputs(pairs_df: pd.DataFrame, delta_g_dict: Dict, delta_m_dict: Dict,
                 niche_adata: sc.AnnData, out_dir: Path, cfg: GridPairingConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pairs_csv = out_dir / "niche_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"\nSaved: {pairs_csv}")
    
    delta_g_npz = out_dir / "niche_delta_g.npz"
    np.savez_compressed(
        delta_g_npz,
        delta_g_keys=np.array(list(delta_g_dict.keys()), dtype=object),
        delta_g_values=np.array(list(delta_g_dict.values()), dtype=object),
        genes=np.array(list(niche_adata.var_names)) if hasattr(niche_adata, "var_names") else None,
    )
    print(f"Saved: {delta_g_npz}")
    
    if delta_m_dict:
        delta_m_npz = out_dir / "niche_delta_m.npz"
        np.savez_compressed(
            delta_m_npz,
            delta_m_keys=np.array(list(delta_m_dict.keys()), dtype=object),
            delta_m_values=np.array(list(delta_m_dict.values()), dtype=object),
        )
        print(f"Saved: {delta_m_npz}")
    
    config_data = {
        "data_type": "grid_niche_pairs",
        "grid_size": cfg.grid_size,
        "n_pairs": len(pairs_df),
        "transitions": pairs_df['transition_tag'].value_counts().to_dict(),
        "has_ot_confidence": 'ot_confidence' in pairs_df.columns,
        "has_delta_m": len(delta_m_dict) > 0,
        "n_delta_m_transitions": len(delta_m_dict),
        "lmdb_path": cfg.lmdb_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    
    if 'ot_confidence' in pairs_df.columns:
        config_data["ot_confidence_mean"] = float(pairs_df['ot_confidence'].mean())
        config_data["ot_confidence_std"] = float(pairs_df['ot_confidence'].std())
    
    with open(out_dir / "niche_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\nTotal pairs: {len(pairs_df)}")
    for tag, count in pairs_df['transition_tag'].value_counts().items():
        print(f"  {tag}: {count}")
    
    if 'ot_confidence' in pairs_df.columns:
        print(f"\nOT Confidence: mean={pairs_df['ot_confidence'].mean():.4f}, "
              f"std={pairs_df['ot_confidence'].std():.4f}")
    
    if delta_m_dict:
        print(f"\nDelta_m signatures saved for {len(delta_m_dict)} transitions")


def main():
    parser = argparse.ArgumentParser(description="SPATIA: Grid-Based Niche Pairing")
    parser.add_argument("--niche_adata", required=True, help="Path to grid niche h5ad")
    parser.add_argument("--cell_adata", required=True, help="Path to cell-level h5ad (for state info)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--lmdb_path", default=None,
                        help="Optional path to grid LMDB for morphology extraction (Δm)")
    parser.add_argument("--dataset_prefix", default="Xenium_FFPE_Human_Breast_Cancer_Rep1_outs",
                        help="Dataset prefix for LMDB keys")
    parser.add_argument("--state_col", default="cell_states", 
                        help="Column in cell_adata.obs containing cell states")
    parser.add_argument("--cell_id_col", default="index",
                        help="Column in cell_adata.obs containing cell IDs that match niche cell_ids")
    parser.add_argument("--grid_size", type=int, default=256)
    parser.add_argument("--min_cells", type=int, default=5)
    parser.add_argument("--min_fraction", type=float, default=0.05)
    parser.add_argument("--max_pairs", type=int, default=200)
    parser.add_argument("--include_ot_confidence", action="store_true",
                        help="Include OT confidence scores in output CSV")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    cfg = GridPairingConfig(
        state_col=args.state_col,
        cell_id_col=args.cell_id_col,
        grid_size=args.grid_size,
        min_cells_per_grid=args.min_cells,
        min_source_fraction=args.min_fraction,
        max_pairs_per_transition=args.max_pairs,
        lmdb_path=args.lmdb_path,
        dataset_prefix=args.dataset_prefix,
        include_ot_confidence=args.include_ot_confidence,
        seed=args.seed,
    )
    
    print("SPATIA Grid-Based Niche Pairing")
    print("=" * 50)
    print(f"Morphology extraction: {'Enabled' if cfg.lmdb_path and MORPHOLOGY_AVAILABLE else 'Disabled'}")
    print(f"OT confidence output: {'Enabled' if cfg.include_ot_confidence else 'Disabled'}")
    if cfg.lmdb_path:
        print(f"  LMDB path: {cfg.lmdb_path}")
    
    print(f"\nLoading niche data: {args.niche_adata}")
    t0 = time.time()
    niche_adata = sc.read_h5ad(args.niche_adata)
    print(f"Niche data shape: {niche_adata.shape} (took {time.time()-t0:.1f}s)")
    
    print(f"\nLoading cell data: {args.cell_adata}")
    t0 = time.time()
    cell_adata = sc.read_h5ad(args.cell_adata)
    print(f"Cell data shape: {cell_adata.shape} (took {time.time()-t0:.1f}s)")
    
    t0 = time.time()
    grid_state_probs = compute_grid_state_composition(cell_adata, niche_adata, cfg)
    print(f"State composition took {time.time()-t0:.1f}s")
    
    t0 = time.time()
    pairs_df, delta_g_dict, delta_m_dict = generate_grid_pairs(niche_adata, grid_state_probs, cfg)
    print(f"Pair generation took {time.time()-t0:.1f}s")
    
    if len(pairs_df) == 0:
        print("\nNo pairs generated!")
        return
    
    save_outputs(pairs_df, delta_g_dict, delta_m_dict, niche_adata, Path(args.out_dir), cfg)


if __name__ == "__main__":
    main()
