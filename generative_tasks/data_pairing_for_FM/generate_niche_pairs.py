
import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


@dataclass
class NichePairingConfig:
    state_col: str = "cell_states"
    niche_col: str = "niche"
    x_col: str = "x_centroid"
    y_col: str = "y_centroid"
    expr_layer: Optional[str] = None
    pool_method: str = "mean"
    pca_dim: int = 50
    sinkhorn_eps: float = 0.05
    sinkhorn_iter: int = 200
    min_cells_per_niche: int = 10
    min_source_fraction: float = 0.05
    max_pairs_per_transition: int = 200
    region_padding_um: float = 50.0
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
                     eps: float, n_iter: int) -> np.ndarray:
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
    
    return matched_indices


def compute_niche_features(
    adata: sc.AnnData,
    niche_id: str,
    cfg: NichePairingConfig,
) -> Optional[Dict]:
    niche_mask = adata.obs[cfg.niche_col] == niche_id
    if niche_mask.sum() < cfg.min_cells_per_niche:
        return None
    
    niche_cells = adata[niche_mask]
    
    X = niche_cells.layers[cfg.expr_layer] if cfg.expr_layer else niche_cells.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    if cfg.pool_method == 'mean':
        pooled_expr = np.mean(X, axis=0)
    else:
        pooled_expr = np.median(X, axis=0)
    
    states = niche_cells.obs[cfg.state_col].values
    state_counts = Counter(states)
    total_cells = len(states)
    state_probs = {state: count / total_cells for state, count in state_counts.items()}
    dominant_states = [s for s, _ in state_counts.most_common(3)]
    
    bbox = None
    centroid = None
    if cfg.x_col in niche_cells.obs.columns and cfg.y_col in niche_cells.obs.columns:
        x_coords = niche_cells.obs[cfg.x_col].values
        y_coords = niche_cells.obs[cfg.y_col].values
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        x_min -= cfg.region_padding_um
        y_min -= cfg.region_padding_um
        x_max += cfg.region_padding_um
        y_max += cfg.region_padding_um
        
        bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
        centroid = (float((x_min + x_max) / 2), float((y_min + y_max) / 2))
    
    return {
        'niche_id': niche_id,
        'n_cells': total_cells,
        'pooled_expr': pooled_expr.astype(np.float32).flatten(),
        'state_probs': state_probs,
        'dominant_states': dominant_states,
        'bbox': bbox,
        'centroid': centroid,
    }


def classify_niche_role(
    features: Dict,
    state_A: str,
    state_B: str,
    min_fraction: float,
) -> str:
    prob_A = features['state_probs'].get(state_A, 0)
    prob_B = features['state_probs'].get(state_B, 0)
    
    if prob_A >= min_fraction and prob_B < min_fraction:
        return 'source'
    elif prob_B >= min_fraction:
        return 'target'
    else:
        return 'other'


def generate_niche_pairs_for_transition(
    adata: sc.AnnData,
    niche_features: Dict[str, Dict],
    state_A: str,
    state_B: str,
    transition_tag: str,
    cfg: NichePairingConfig,
    rng: np.random.Generator,
) -> Tuple[List[Dict], List[np.ndarray]]:
    
    source_niches = []
    target_niches = []
    
    for niche_id, features in niche_features.items():
        role = classify_niche_role(features, state_A, state_B, cfg.min_source_fraction)
        if role == 'source':
            source_niches.append(niche_id)
        elif role == 'target':
            target_niches.append(niche_id)
    
    print(f"    Source niches ({state_A}-enriched): {len(source_niches)}")
    print(f"    Target niches ({state_B}-enriched): {len(target_niches)}")
    
    if len(source_niches) < 2 or len(target_niches) < 2:
        print(f"    Skipping: insufficient niches")
        return [], []
    
    if len(source_niches) > cfg.max_pairs_per_transition:
        source_niches = list(rng.choice(source_niches, cfg.max_pairs_per_transition, replace=False))
    if len(target_niches) > cfg.max_pairs_per_transition:
        target_niches = list(rng.choice(target_niches, cfg.max_pairs_per_transition, replace=False))
    
    source_expr = np.vstack([niche_features[n]['pooled_expr'] for n in source_niches])
    target_expr = np.vstack([niche_features[n]['pooled_expr'] for n in target_niches])
    
    combined = np.vstack([source_expr, target_expr])
    combined_pca = pca_project(combined, cfg.pca_dim)
    source_pca = combined_pca[:len(source_niches)]
    target_pca = combined_pca[len(source_niches):]
    
    matched_indices = sinkhorn_pairing(source_pca, target_pca, cfg.sinkhorn_eps, cfg.sinkhorn_iter)
    
    pair_records = []
    delta_g_vectors = []
    
    for i, j in enumerate(matched_indices):
        src_id = source_niches[i]
        tgt_id = target_niches[j]
        src_feat = niche_features[src_id]
        tgt_feat = niche_features[tgt_id]
        
        record = {
            'source_niche_id': src_id,
            'target_niche_id': tgt_id,
            'source_n_cells': src_feat['n_cells'],
            'target_n_cells': tgt_feat['n_cells'],
            'source_state_A_frac': src_feat['state_probs'].get(state_A, 0),
            'source_state_B_frac': src_feat['state_probs'].get(state_B, 0),
            'target_state_A_frac': tgt_feat['state_probs'].get(state_A, 0),
            'target_state_B_frac': tgt_feat['state_probs'].get(state_B, 0),
            'source_dominant_states': '|'.join(src_feat['dominant_states']),
            'target_dominant_states': '|'.join(tgt_feat['dominant_states']),
            'state_A': state_A,
            'state_B': state_B,
            'transition_tag': transition_tag,
        }
        
        if src_feat['bbox']:
            record['source_bbox_xmin'] = src_feat['bbox'][0]
            record['source_bbox_ymin'] = src_feat['bbox'][1]
            record['source_bbox_xmax'] = src_feat['bbox'][2]
            record['source_bbox_ymax'] = src_feat['bbox'][3]
        if tgt_feat['bbox']:
            record['target_bbox_xmin'] = tgt_feat['bbox'][0]
            record['target_bbox_ymin'] = tgt_feat['bbox'][1]
            record['target_bbox_xmax'] = tgt_feat['bbox'][2]
            record['target_bbox_ymax'] = tgt_feat['bbox'][3]
        if src_feat['centroid']:
            record['source_centroid_x'] = src_feat['centroid'][0]
            record['source_centroid_y'] = src_feat['centroid'][1]
        if tgt_feat['centroid']:
            record['target_centroid_x'] = tgt_feat['centroid'][0]
            record['target_centroid_y'] = tgt_feat['centroid'][1]
        
        pair_records.append(record)
        
        delta = tgt_feat['pooled_expr'] - src_feat['pooled_expr']
        delta_g_vectors.append(delta)
    
    print(f"    Generated {len(pair_records)} niche pairs")
    return pair_records, delta_g_vectors


def run_niche_pairing_pipeline(
    adata: sc.AnnData,
    cfg: NichePairingConfig,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    
    rng = np.random.default_rng(cfg.seed)
    
    for col in [cfg.state_col, cfg.niche_col]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")
    
    print(f"\nComputing features for {adata.obs[cfg.niche_col].nunique()} niches...")
    niche_features = {}
    for niche_id in tqdm(adata.obs[cfg.niche_col].unique(), desc="Niches"):
        features = compute_niche_features(adata, niche_id, cfg)
        if features:
            niche_features[niche_id] = features
    
    print(f"Valid niches (>= {cfg.min_cells_per_niche} cells): {len(niche_features)}")
    
    all_pairs = []
    transition_delta_g: Dict[str, List[np.ndarray]] = {}
    
    available_states = set(adata.obs[cfg.state_col].unique())
    
    for state_A, state_B, tag, task in BIOLOGICAL_TRANSITIONS:
        print(f"\nProcessing {tag} ({state_A} → {state_B}):")
        
        if state_A not in available_states or state_B not in available_states:
            print(f"    States not found in data")
            continue
        
        pairs, deltas = generate_niche_pairs_for_transition(
            adata, niche_features, state_A, state_B, tag, cfg, rng
        )
        
        if pairs:
            for p in pairs:
                p['task_name'] = task
            all_pairs.extend(pairs)
            
            if tag not in transition_delta_g:
                transition_delta_g[tag] = []
            transition_delta_g[tag].extend(deltas)
    
    if not all_pairs:
        print("\nNo niche pairs generated!")
        return pd.DataFrame(), {}
    
    pairs_df = pd.DataFrame(all_pairs)
    
    delta_g_dict = {}
    for tag, deltas in transition_delta_g.items():
        if deltas:
            stacked = np.vstack(deltas)
            delta_g_dict[tag] = np.mean(stacked, axis=0).astype(np.float32)
    
    return pairs_df, delta_g_dict


def save_niche_outputs(
    pairs_df: pd.DataFrame,
    delta_g_dict: Dict[str, np.ndarray],
    adata: sc.AnnData,
    out_dir: Path,
    cfg: NichePairingConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pairs_csv = out_dir / "niche_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"\nSaved: {pairs_csv}")
    
    delta_npz = out_dir / "niche_delta_g.npz"
    np.savez_compressed(
        delta_npz,
        delta_g_keys=np.array(list(delta_g_dict.keys()), dtype=object),
        delta_g_values=np.array(list(delta_g_dict.values()), dtype=object),
        genes=np.array(list(adata.var_names)) if hasattr(adata, "var_names") else None,
    )
    print(f"Saved: {delta_npz}")
    
    config_data = {
        "data_type": "niche_level_pairs",
        "description": "Niche/region level OT pairs for spatial biological transitions",
        "n_pairs": len(pairs_df),
        "transitions": pairs_df['transition_tag'].value_counts().to_dict(),
        "tasks": pairs_df['task_name'].value_counts().to_dict() if 'task_name' in pairs_df.columns else {},
        "parameters": {
            "pool_method": cfg.pool_method,
            "pca_dim": cfg.pca_dim,
            "min_cells_per_niche": cfg.min_cells_per_niche,
            "min_source_fraction": cfg.min_source_fraction,
            "region_padding_um": cfg.region_padding_um,
            "seed": cfg.seed,
        },
        "avg_source_niche_size": float(pairs_df['source_n_cells'].mean()),
        "avg_target_niche_size": float(pairs_df['target_n_cells'].mean()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    
    config_json = out_dir / "niche_config.json"
    with open(config_json, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved: {config_json}")
    
    print(f"\n{'='*50}")
    print("NICHE-LEVEL PAIRING SUMMARY")
    print(f"{'='*50}")
    print(f"Total pairs: {len(pairs_df)}")
    print(f"\nPairs per transition:")
    for tag, count in pairs_df['transition_tag'].value_counts().items():
        print(f"  {tag}: {count}")
    print(f"\nAverage niche sizes:")
    print(f"  Source: {pairs_df['source_n_cells'].mean():.0f} cells")
    print(f"  Target: {pairs_df['target_n_cells'].mean():.0f} cells")
    
    if 'source_bbox_xmin' in pairs_df.columns:
        src_widths = pairs_df['source_bbox_xmax'] - pairs_df['source_bbox_xmin']
        tgt_widths = pairs_df['target_bbox_xmax'] - pairs_df['target_bbox_xmin']
        print(f"\nAverage region sizes (um):")
        print(f"  Source: {src_widths.mean():.1f} × {(pairs_df['source_bbox_ymax'] - pairs_df['source_bbox_ymin']).mean():.1f}")
        print(f"  Target: {tgt_widths.mean():.1f} × {(pairs_df['target_bbox_ymax'] - pairs_df['target_bbox_ymin']).mean():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="SPATIA: Niche-Level OT Pairing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--adata", required=True, help="Path to AnnData .h5ad file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--state_col", default="cell_states", help="Column for cell states")
    parser.add_argument("--niche_col", default="niche", help="Column for niche IDs")
    parser.add_argument("--x_col", default="x_centroid", help="Column for x coordinates")
    parser.add_argument("--y_col", default="y_centroid", help="Column for y coordinates")
    parser.add_argument("--pool_method", default="mean", choices=['mean', 'median'])
    parser.add_argument("--min_cells", type=int, default=10, help="Min cells per niche")
    parser.add_argument("--min_fraction", type=float, default=0.05, help="Min source state fraction")
    parser.add_argument("--max_pairs", type=int, default=200, help="Max pairs per transition")
    parser.add_argument("--padding_um", type=float, default=50.0, help="Region padding in microns")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    cfg = NichePairingConfig(
        state_col=args.state_col,
        niche_col=args.niche_col,
        x_col=args.x_col,
        y_col=args.y_col,
        pool_method=args.pool_method,
        min_cells_per_niche=args.min_cells,
        min_source_fraction=args.min_fraction,
        max_pairs_per_transition=args.max_pairs,
        region_padding_um=args.padding_um,
        seed=args.seed,
    )
    
    print("SPATIA Niche-Level Pairing Pipeline")
    print("=" * 50)
    print(f"Input: {args.adata}")
    print(f"Output: {args.out_dir}")
    
    print(f"\nLoading AnnData...")
    adata = sc.read_h5ad(args.adata)
    print(f"Shape: {adata.shape}")
    print(f"Niches: {adata.obs[cfg.niche_col].nunique()}")
    
    pairs_df, delta_g_dict = run_niche_pairing_pipeline(adata, cfg)
    
    if len(pairs_df) == 0:
        print("\nNo pairs generated. Check niche/state columns.")
        return
    
    save_niche_outputs(pairs_df, delta_g_dict, adata, Path(args.out_dir), cfg)


if __name__ == "__main__":
    main()
