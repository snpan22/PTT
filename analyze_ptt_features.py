"""
analyze_ptt_features.py
=======================
Analysis and visualization for PTT instrumentation output.

Loads HDF5 files from eval_ptt_instrumented.py and produces:
  1. Feature L2-norm distributions by proposal category (violin plots)
  2. t-SNE / PCA scatter coloured by category
  3. Attention weight heatmap grids (sampled proposals per category)
  4. P-P displacement trajectory plots (box_seq over time)
  5. Attention entropy vs. k (spoofing-duration saturation curve)
  6. Feature norm vs. k for spoofed proposals (per-layer trajectory)

USAGE
-----
    python analyze_ptt_features.py \\
        --features_dir /path/to/feature_output_dir \\
        --out_dir      /path/to/analysis_output \\
        --max_proposals 50000
"""

import argparse
import glob
import os

import h5py
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES  = ['real_tp', 'spoofed_fp', 'clean_fp']
CAT_COLORS  = {'real_tp': '#2196F3', 'spoofed_fp': '#F44336', 'clean_fp': '#4CAF50'}
CAT_LABELS  = {'real_tp': 'Real TP', 'spoofed_fp': 'Spoofed FP', 'clean_fp': 'Clean FP'}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(features_dir, feature_keys, max_per_cat=50000, k_filter=None):
    """
    Walk all *_features.h5 files and collect feature arrays grouped by category.

    Parameters
    ----------
    feature_keys : list[str]   e.g. ['long_mem', 'cur_feat_final']
    max_per_cat  : int         memory guard — cap per category
    k_filter     : int | None  if set, only load frames with this exact k_value

    Returns
    -------
    data : dict[cat -> dict[key -> np.ndarray(N, ...)]]
    meta : dict[cat -> dict[field -> np.ndarray(N,)]]
           fields: confidence, iou_real, iou_spoof, cls_logit, k_value
    """
    data   = {cat: {k: [] for k in feature_keys} for cat in CATEGORIES}
    meta   = {cat: {'confidence': [], 'iou_real': [], 'iou_spoof': [],
                    'cls_logit': [], 'k_value': []}
              for cat in CATEGORIES}
    counts = {cat: 0 for cat in CATEGORIES}

    #find h5 files in directory and loop through them
    h5_files = sorted(glob.glob(os.path.join(features_dir, '*_features.h5')))
    for h5_path in tqdm(h5_files, desc="Loading HDF5"):
        try:
            with h5py.File(h5_path, 'r') as hf:
                
                #
                if 'frames' not in hf:
                    continue
                
                #loop thru frames in h5 file
                for frame_id in hf['frames']:
                    frame_grp = hf['frames'][frame_id]
                    
                    #optional filtering by k value (number of spoofed frames in history)
                    k_val = int(frame_grp.attrs.get('k_value', -1))
                    if k_filter is not None and k_val != k_filter:
                        continue
                    
                    #access proposal group
                    pgrp = frame_grp.get('proposals', None)
                    if pgrp is None or 'category' not in pgrp:
                        continue

                    #categories stored as byte strings. convert backt to Python strings like "real_tp" or "spoofed_fp"
                    cats_raw = pgrp['category'][:]
                    cats = np.array(
                        [c.decode() if isinstance(c, bytes) else c for c in cats_raw]
                    )
                    
                    #scalar data... one per proposal
                    conf  = pgrp['confidence'][:] if 'confidence' in pgrp else None
                    iou_r = pgrp['iou_real'][:]   if 'iou_real'   in pgrp else None
                    iou_s = pgrp['iou_spoof'][:]  if 'iou_spoof'  in pgrp else None
                    cl    = pgrp['cls_logit'][:]  if 'cls_logit'  in pgrp else None

                    # Loop through categories and append data, respecting max_per_cat limit
                    for cat in CATEGORIES:
                        if counts[cat] >= max_per_cat:
                            continue
                        
                        #select only proposals of this category
                        mask = (cats == cat)
                        if not mask.any():
                            continue
                        
                        # Trim to remaining budget (memory cap)
                        remaining = max_per_cat - counts[cat]
                        hit_idxs  = np.where(mask)[0]
                        if len(hit_idxs) > remaining:
                            
                            #truncate selection to first remaining indices if there are more proposals than remaining budget allows
                            hit_idxs = hit_idxs[:remaining]
                            mask = np.zeros(len(cats), dtype=bool)
                            mask[hit_idxs] = True

                        #load requested feature tensors
                        for fk in feature_keys:
                            if fk not in pgrp:
                                continue
                            data[cat][fk].append(pgrp[fk][mask])

                        #collect scalar metadata
                        if conf  is not None: meta[cat]['confidence'].extend(conf[mask])
                        if iou_r is not None: meta[cat]['iou_real'].extend(iou_r[mask])
                        if iou_s is not None: meta[cat]['iou_spoof'].extend(iou_s[mask])
                        if cl    is not None: meta[cat]['cls_logit'].extend(cl[mask])
                        meta[cat]['k_value'].extend([k_val] * mask.sum())
                        counts[cat] += mask.sum()
        except Exception as e:
            print(f"Warning: skipping {h5_path}: {e}")

    # Concatenate each list into one array
    for cat in CATEGORIES:
        for fk in feature_keys:
            arrs = [a for a in data[cat][fk] if a is not None and len(a) > 0]
            data[cat][fk] = np.concatenate(arrs, axis=0) if arrs else np.array([])
        for mk in meta[cat]:
            meta[cat][mk] = np.array(meta[cat][mk])

    print("Proposals loaded:")
    for cat in CATEGORIES:
        print(f"  {cat:15s}: {counts[cat]:6d}")

    return data, meta


# ---------------------------------------------------------------------------
# Analysis 1: Feature L2-norm distributions (violin plots)
# ---------------------------------------------------------------------------

def plot_norm_distributions(data, feature_key, out_path, log_scale=True):
    """Violin plot of per-proposal L2 norms, one violin per category.

       feature_key; which feature to analyze
       out_path: where to save plot
       
       How strong (in magnitude) are internal features for each category?
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = list(range(len(CATEGORIES)))
    all_norms = []
    present   = []

    #loop over categories
    for i, cat in enumerate(CATEGORIES):
        
        #feature tensor for each category
        arr = data[cat].get(feature_key, np.array([]))
        if arr is None or len(arr) == 0:
            all_norms.append(None)
            continue
        
        # convert to 1D vector
        flat  = arr.reshape(len(arr), -1).astype(np.float64)
        
        #L2 norm
        norms = np.linalg.norm(flat, axis=1)
        if log_scale:
            norms = np.log1p(norms)
        all_norms.append(norms)
        present.append(i)

    valid_norms = [all_norms[i] for i in present]
    valid_pos   = [positions[i] for i in present]

    if not valid_norms:
        plt.close()
        return

    parts = ax.violinplot(valid_norms, positions=valid_pos, showmedians=True)
    for pc, idx in zip(parts['bodies'], present):
        pc.set_facecolor(CAT_COLORS[CATEGORIES[idx]])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES])
    ax.set_ylabel('log(1 + L2 norm)' if log_scale else 'L2 norm')
    ax.set_title(f'Feature norm distribution — {feature_key}')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis 2: t-SNE / PCA scatter
# ---------------------------------------------------------------------------

def plot_tsne(data, feature_key, out_path, max_pts=3000):
    """2D t-SNE scatter of proposal features coloured by category.
        
        feature_key: which features to visualize
        
        how different proposal categories are seperated in feature space
        - project high dimensional features into 2D using t-SNE   
    """
    if not HAS_SKLEARN:
        print("sklearn not available — skipping t-SNE")
        return


    all_feats, all_cats = [], []
    
    #obtain feature tensors for each category
    for cat in CATEGORIES:
        arr = data[cat].get(feature_key, np.array([]))
        if arr is None or len(arr) == 0:
            continue
        #flatten features into 1D feature vecto
        flat = arr.reshape(len(arr), -1).astype(np.float32)
        
        #subsample: limit number of points per category
        if len(flat) > max_pts:
            idx  = np.random.choice(len(flat), max_pts, replace=False)
            flat = flat[idx]
        all_feats.append(flat)
        all_cats.extend([cat] * len(flat))

    if not all_feats:
        return

    #combine into one matrix
    X        = np.concatenate(all_feats, axis=0)
    cats_arr = np.array(all_cats)

    #normalize: zero mean, unit variance per dimension
    X = StandardScaler().fit_transform(X)
    
    #optional PCA: reduces very high dimensional features to 50D
    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)
        
    
    emb = TSNE(n_components=2, random_state=42, perplexity=40,
               n_iter=1000).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 7))
    for cat in CATEGORIES:
        mask = (cats_arr == cat)
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CAT_COLORS[cat], label=CAT_LABELS[cat],
                   s=5, alpha=0.6)
    ax.legend(markerscale=3)
    ax.set_title(f't-SNE — {feature_key}')
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis 3: Attention heatmap grids
# ---------------------------------------------------------------------------

def plot_attention_heatmaps(features_dir, attn_key, out_path, n_examples=6):
    """
    Sample n_examples proposals of each category and plot their attention
    matrices side-by-side. Reads directly from HDF5 (no full data load).
    
    visualize raw attention matrices from HDF5 files by sampling few proposals per category and plotting attention weights as heatmaps
    """
    examples = {cat: [] for cat in CATEGORIES}
    h5_files = sorted(glob.glob(os.path.join(features_dir, '*_features.h5')))

    for h5_path in h5_files:
        if all(len(v) >= n_examples for v in examples.values()):
            break
        try:
            with h5py.File(h5_path, 'r') as hf:
                if 'frames' not in hf:
                    continue
                for frame_id in hf['frames']:
                    if all(len(v) >= n_examples for v in examples.values()):
                        break
                    pgrp = hf['frames'][frame_id].get('proposals', None)
                    if pgrp is None or attn_key not in pgrp:
                        continue
                    cats_raw   = pgrp['category'][:]
                    cats       = np.array([c.decode() if isinstance(c, bytes)
                                           else c for c in cats_raw])
                    attn_mats  = pgrp[attn_key][:]   # (R, Q, K)
                    for cat in CATEGORIES:
                        if len(examples[cat]) >= n_examples:
                            continue
                        for i in np.where(cats == cat)[0]:
                            examples[cat].append(attn_mats[i])
                            if len(examples[cat]) >= n_examples:
                                break
        except Exception:
            continue

    n_rows = sum(1 for v in examples.values() if v)
    if n_rows == 0:
        print(f"No examples found for {attn_key}")
        return

    fig, axes = plt.subplots(
        n_rows, n_examples, figsize=(n_examples * 2, n_rows * 2.5)
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    row = 0
    for cat in CATEGORIES:
        if not examples[cat]:
            continue
        for col in range(n_examples):
            ax = axes[row, col]
            if col < len(examples[cat]):
                mat = examples[cat][col]
                ax.imshow(mat, vmin=0, vmax=mat.max(), cmap='hot', aspect='auto')
                if col == 0:
                    ax.set_ylabel(CAT_LABELS[cat], fontsize=9)
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
        row += 1

    fig.suptitle(f'Attention heatmaps — {attn_key}', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis 4: P-P displacement trajectory plots
# ---------------------------------------------------------------------------

def plot_pp_trajectories(data, out_path, n_examples=10):
    """
    Plot XY displacement magnitude from box_seq over time.
    box_seq shape per proposal: (8, T_total) — 8 features over T frames.
    Features[0]=dx, [1]=dy relative to current position.
    
    visualise how proposal positions evolve over time by plotting displacement magnitude across proposal history
    
    How stable or consistent is motion history of proposals?
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, cat in zip(axes, ['real_tp', 'spoofed_fp']):
        arr = data[cat].get('box_seq', np.array([]))
        if arr is None or len(arr) == 0:
            ax.text(0.5, 0.5, f'No {cat} data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(CAT_LABELS.get(cat, cat))
            continue

        idx = np.random.choice(len(arr), min(n_examples, len(arr)), replace=False)
        for i in idx:
            seq = arr[i]              # (8, T_total)
            T   = seq.shape[1]
            frames = np.arange(T)
            dx   = seq[0, ::-1]       # reverse so index 0 = most recent
            dy   = seq[1, ::-1]
            disp = np.sqrt(dx**2 + dy**2)
            ax.plot(frames, disp, alpha=0.5, linewidth=1, color=CAT_COLORS[cat])

        ax.set_xlabel('Frame index (0 = current)')
        ax.set_ylabel('XY displacement magnitude')
        ax.set_title(CAT_LABELS.get(cat, cat))
        ax.grid(alpha=0.3)

    fig.suptitle('P-P displacement features over historical frames')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis 5: Attention entropy vs. k
# ---------------------------------------------------------------------------

def plot_attention_entropy_by_k(features_dir, out_path, max_frames=5000):
    """
    For each k value (spoofing duration), compute mean entropy of the
    long-term self-attention matrix attn_long for spoofed and real proposals.

    High entropy = flat/uniform attention = saturation.
    
    - how focused or diffuse model's long term attention becomes as spoofing persists over time
    - if spoofed entropy increases with k: attention becomes more uniform, strong evidence of temporal confusion
    - if spoofed FP entropy ~= real TP: model treatses spoofed trajectories like real ones
    - if spoofed FP entropy > real TP: model is uncertain about spoof
    - if entropy decreases: model locking onto spoof over time
    
    Does longer spoof duration corrupt temporal attention?
    """
    k_entropies = {cat: {} for cat in CATEGORIES}
    h5_files    = sorted(glob.glob(os.path.join(features_dir, '*_features.h5')))
    frame_count = 0

    for h5_path in h5_files:
        if frame_count >= max_frames:
            break
        try:
            with h5py.File(h5_path, 'r') as hf:
                if 'frames' not in hf:
                    continue
                for frame_id in hf['frames']:
                    frame_grp = hf['frames'][frame_id]
                    k_val = int(frame_grp.attrs.get('k_value', -1))
                    if k_val < 0:
                        continue
                    pgrp = frame_grp.get('proposals', None)
                    if pgrp is None or 'attn_long' not in pgrp:
                        continue

                    cats_raw  = pgrp['category'][:]
                    cats      = np.array([c.decode() if isinstance(c, bytes)
                                          else c for c in cats_raw])
                    attn_mats = pgrp['attn_long'][:]   # (R, Tl, Tl)

                    for cat in CATEGORIES:
                        mask = (cats == cat)
                        if not mask.any():
                            continue
                        mats = attn_mats[mask].astype(np.float64)   # (N, Tl, Tl)
                        # Per-row entropy, averaged over all rows and proposals
                        eps = 1e-9
                        ent = -(mats * np.log(mats + eps)).sum(axis=-1).mean()
                        k_entropies[cat].setdefault(k_val, []).append(float(ent))
                    frame_count += 1
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in ['spoofed_fp', 'real_tp', 'clean_fp']:
        kv = k_entropies[cat]
        if not kv:
            continue
        ks    = sorted(kv.keys())
        means = np.array([np.mean(kv[k]) for k in ks])
        stds  = np.array([np.std(kv[k])  for k in ks])
        ax.plot(ks, means, color=CAT_COLORS[cat],
                label=CAT_LABELS[cat], linewidth=2)
        ax.fill_between(ks, means - stds, means + stds,
                         color=CAT_COLORS[cat], alpha=0.15)

    ax.set_xlabel('k  (number of spoofed frames in history)')
    ax.set_ylabel('Mean attention entropy  (long-term self-attn)')
    ax.set_title('Attention saturation vs. spoofing duration\n'
                 'Rising entropy = increasingly uniform / saturated attention')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Analysis 6: Feature norm vs. k for spoofed proposals
# ---------------------------------------------------------------------------

def plot_feature_norms_by_k(features_dir, feature_key, out_path,
                             max_frames=5000):
    """
    Track how the L2 norm of a feature evolves as k increases for spoofed
    proposals.  Shows which internal representations are most affected.
    
    For spoofed false positives only: compute feature ("long mem", "cur_feat_final", etc) L2 norm vs k (spoof length)
    - if feature norm increases with k: spoofed object is getting strtonger internal representation, temporal memory is reinforcing it
    - if feature norm saturates: model has reached steady belief about spoof
    - if feature norm decreases: model is discounting spoof over time, temporal reasomning is helping reject it
    
    Does repeated spoofing strengthen model's belief?
    """
    k_norms     = {}
    h5_files    = sorted(glob.glob(os.path.join(features_dir, '*_features.h5')))
    frame_count = 0

    for h5_path in h5_files:
        if frame_count >= max_frames:
            break
        try:
            with h5py.File(h5_path, 'r') as hf:
                if 'frames' not in hf:
                    continue
                for frame_id in hf['frames']:
                    frame_grp = hf['frames'][frame_id]
                    k_val = int(frame_grp.attrs.get('k_value', -1))
                    if k_val < 1:
                        continue
                    pgrp = frame_grp.get('proposals', None)
                    if pgrp is None or feature_key not in pgrp:
                        continue
                    cats_raw = pgrp['category'][:]
                    cats     = np.array([c.decode() if isinstance(c, bytes)
                                         else c for c in cats_raw])
                    mask     = (cats == 'spoofed_fp')
                    if not mask.any():
                        continue
                    feats = pgrp[feature_key][mask]
                    norms = np.linalg.norm(
                        feats.reshape(len(feats), -1).astype(np.float64), axis=1
                    )
                    k_norms.setdefault(k_val, []).extend(norms.tolist())
                    frame_count += 1
        except Exception:
            continue

    if not k_norms:
        print(f"No data for {feature_key} norm-vs-k plot")
        return

    ks    = sorted(k_norms.keys())
    means = np.array([np.mean(k_norms[k]) for k in ks])
    stds  = np.array([np.std(k_norms[k])  for k in ks])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ks, means, color='#F44336', linewidth=2)
    ax.fill_between(ks, means - stds, means + stds, alpha=0.2, color='#F44336')
    ax.set_xlabel('k  (spoofed frames in history)')
    ax.set_ylabel('Mean L2 norm  (spoofed proposals)')
    ax.set_title(f'{feature_key} norm vs. spoofing duration')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--features_dir', required=True,
                   help='Dir with *_features.h5 from eval_ptt_instrumented')
    p.add_argument('--out_dir',      required=True)
    p.add_argument('--max_proposals', type=int, default=100000)
    p.add_argument('--no_tsne',      action='store_true',
                   help='Skip t-SNE (slow for large datasets)')
    p.add_argument('--seed',         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading feature data...")
    feature_keys = [
        'cur_feat_final', 'long_mem', 'short_mem', 'future_feat',
        'enhanced_pt', 'cur_feat_sa',  'box_seq',
    ]
    data, meta = load_features(
        args.features_dir, feature_keys, max_per_cat=args.max_proposals
    )

    # --- 1. Norm distributions ---
    for fk in ['cur_feat_final', 'long_mem', 'short_mem', 
               'enhanced_pt']:
        plot_norm_distributions(
            data, fk,
            os.path.join(args.out_dir, f'norm_dist_{fk}.png'),
        )

    # --- 2. t-SNE ---
    if not args.no_tsne and HAS_SKLEARN:
        for fk in ['cur_feat_final', 'long_mem']:
            plot_tsne(
                data, fk,
                os.path.join(args.out_dir, f'tsne_{fk}.png'),
            )
    elif not HAS_SKLEARN:
        print("sklearn not found — install scikit-learn to enable t-SNE plots")

    # --- 3. Attention heatmaps ---
    for ak in ['attn_long', 'attn_short', 'attn2', 'cross_long_w']:
        plot_attention_heatmaps(
            args.features_dir, ak,
            os.path.join(args.out_dir, f'attn_heatmap_{ak}.png'),
        )

    # --- 4. P-P trajectory plots ---
    plot_pp_trajectories(
        data,
        os.path.join(args.out_dir, 'pp_trajectories.png'),
    )

    # --- 5. Attention entropy vs. k ---
    plot_attention_entropy_by_k(
        args.features_dir,
        os.path.join(args.out_dir, 'attn_entropy_by_k.png'),
    )

    # --- 6. Feature norm vs. k ---
    for fk in ['cur_feat_final', 'long_mem', 'enhanced_pt', 'short_mem']:
        plot_feature_norms_by_k(
            args.features_dir, fk,
            os.path.join(args.out_dir, f'norm_vs_k_{fk}.png'),
        )

    print(f"\nAll plots saved to {args.out_dir}")


if __name__ == '__main__':
    main()
