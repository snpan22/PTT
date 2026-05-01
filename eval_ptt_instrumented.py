"""
eval_ptt_instrumented.py
========================
Inference script that runs PTT with activation hooks enabled and saves
per-proposal feature records to HDF5.

Mirrors the forward-pass loop of eval_ptt_final.py and adds instrumentation
via PTTInstrumentationContext.  One HDF5 file is produced per segment.

OUTPUT HDF5 STRUCTURE  (<out_dir>/<segment>_features.h5)
---------------------------------------------------------
attrs:  segment (str), num_frames (int)
frames/<frame_id>/
  attrs: frame_id, seg_idx, k_value
  proposals/
    attrs: num_proposals
    category          (R,)      byte str  real_tp|spoofed_fp|clean_fp|background
    confidence        (R,)      f32
    roi_box           (R, 7)    f32
    iou_real          (R,)      f32
    iou_spoof         (R,)      f32
    cls_logit         (R,)      f32
    geometry_feat     (R,T*N,D) f32   G^t per-frame geometry features
    long_mem          (R,Tl,96) f32   M-hat_l long-term memory
    short_mem         (R,Ts,96) f32   M-hat_s short-term memory
    future_feat       (R,Tf,96) f32   M-hat_f future features
    enhanced_pt       (R,T,128) f32   enhanced P-T features
    cur_feat_sa       (R,D)     f32   Ghat^T after self-attn + pool
    cross_long        (R,1,256) f32   M-hat_lp aggregator output
    cross_short       (R,1,256) f32   M-hat_sp aggregator output
    cross_future      (R,1,256) f32   M-hat_fp aggregator output
    cur_feat_final    (R,256)   f32   Ghat'^T after temporal aggregation
    box_seq           (R,8,T)   f32   P-P displacement features
    attn_long         (R,Tl,Tl) f32   [capture_attn only]
    attn_short        (R,Ts,Tl) f32
    attn_future       (R,Tf,Tsl)f32
    attn2             (R,T,T)   f32
    cross_long_w      (R,1,Tl)  f32
    cross_short_w     (R,1,Ts)  f32
    cross_future_w    (R,1,Tf)  f32

USAGE
-----
    python eval_ptt_instrumented.py \\
        --cfg_file   tools/cfgs/waymo_models/ptt_32frames.yaml \\
        --ckpt       /path/to/checkpoint.pth \\
        --dataset    /path/to/spoofed_dataset_dir \\
        --timing_annos /path/to/timing_annos_dir \\
        --out_dir    /path/to/feature_output_dir \\
        --capture_attn
"""

import argparse
import logging
import os
import pickle as pkl
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils

import importlib
import helpers_ptt
importlib.reload(helpers_ptt)

from ptt_instrumentation import PTTInstrumentationContext, collect_proposal_features


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path):
    logger = logging.getLogger("eval_ptt_instrumented")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in [logging.FileHandler(log_path), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file',     required=True)
    p.add_argument('--ckpt',         required=True)
    p.add_argument('--dataset',      required=True,
                   help='Dir with <seg>_d.pkl spoofed dataset files')
    p.add_argument('--timing_annos', required=True,
                   help='Dir with <seg>_a.pkl timing annotation files')
    p.add_argument('--detector',     default='ptt', choices=['ptt', 'msf'])
    p.add_argument('--out_dir',      required=True)
    p.add_argument('--log_file',     default='eval_ptt_instrumented.log')
    p.add_argument('--workers',      type=int, default=4)
    p.add_argument('--capture_attn', action='store_true',
                   help='Also save attention weight matrices (larger output files)')
    p.add_argument('--max_segments', type=int, default=None,
                   help='Limit to first N segments (for debugging)')
    p.add_argument('--history',      type=int, default=32,
                   help='Number of historical frames fed to PTT')
    return p.parse_args()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# ---------------------------------------------------------------------------
# Timing key -> k_value mapping
# ---------------------------------------------------------------------------



def build_timing_lookup(timing_arr):
    """
    Returns dict: local_frame_idx -> k_value where
      k=0 : clean frame just before a spoof block  (timing[idx-1]==0, timing[idx]==1)
      k=1 : first frame of a spoof block           (timing==1)
      k=2..32: subsequent frames inside the block  (timing==2)
    Frames not near any block map to None.
    """
    lookup = {}
    n = len(timing_arr)
    block_starts = [i for i, t in enumerate(timing_arr) if t == 1]
    for bs in block_starts:
        k0 = bs - 1
        if k0 >= 0 and timing_arr[k0] == 0:
            lookup[k0] = 0
        for k in range(1, 34):
            offset = bs + (k - 1)
            if offset >= n:
                break
            expected = 1 if k == 1 else 2
            if timing_arr[offset] != expected:
                break
            lookup[offset] = k
    return lookup


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_frame_to_hdf5(hf, frame_id, seg_idx, k_value, records):
    """Write per-proposal records for one frame into an open HDF5 file."""
    if not records:
        return

    #creates/reuses group for this frame inside the file
    grp = hf.require_group(f"frames/{frame_id}")
    
    #attach metadata
    grp.attrs['frame_id']    = np.string_(frame_id)
    grp.attrs['seg_idx']     = seg_idx
    grp.attrs['k_value']     = k_value if k_value is not None else -1
    #create a proposals/ group, where all per-proposal data goes
    pgrp = grp.require_group("proposals")
    num_rois = len(records)
    pgrp.attrs['num_proposals'] = num_rois

    # Scalar per-proposal arrays
    #takes one value per proposal and stack into arrays R times, R = # proposals
    pgrp.create_dataset(
        'category',
        data=np.array([r['category'] for r in records], dtype='S16'),
    )
    pgrp.create_dataset(
        'confidence',
        data=np.array([r['confidence'] for r in records], dtype=np.float32),
    )
    pgrp.create_dataset(
        'iou_real',
        data=np.array([r['iou_real'] for r in records], dtype=np.float32),
    )
    pgrp.create_dataset(
        'iou_spoof',
        data=np.array([r['iou_spoof'] for r in records], dtype=np.float32),
    )
    pgrp.create_dataset(
        'cls_logit',
        data=np.array(
            [r['cls_logit'] if r['cls_logit'] is not None else np.nan
             for r in records],
            dtype=np.float32,
        ),
    )
    pgrp.create_dataset('roi_box', data=np.stack([r['roi_box'] for r in records]))

    # Multidimensional feature arrays - stack across proposals
    #loop thru internal features
    feature_keys = [
        # geometry_feat omitted: shape (R, T*N, D) ≈ 2 MB/proposal, too large;
        # processed features downstream are more informative for vulnerability analysis.
        'long_mem', 'short_mem', 'future_feat', 'enhanced_pt',
        'cur_feat_sa', 'cross_long', 'cross_short', 'cross_future',
        'cur_feat_final', 'feat_traj', 'box_seq',
        'attn_long', 'attn_short', 'attn_future', 'attn2',
        'cross_long_w', 'cross_short_w', 'cross_future_w',
    ]
    #for each feature collect feature from every proposal, check if exists and is consistent
    for fkey in feature_keys:
        arrays = [r[fkey] for r in records]
        if all(a is None for a in arrays):
            continue
        # Skip if any proposal is missing this feature (ragged would be invalid)
        if any(a is None for a in arrays):
            continue
        try:
            #converts list of per-proposal tensors into single array
            stacked = np.stack(arrays, axis=0).astype(np.float32)
            pgrp.create_dataset(
                fkey, data=stacked, compression='gzip', compression_opts=4
            )
        except Exception as e:
            pass   # skip features with inconsistent shapes silently


# ---------------------------------------------------------------------------
# Proposal list builder (mirrors eval_ptt_final.py logic)
# ---------------------------------------------------------------------------

# def build_proposals_tensor(
#     seg_dataset, seg_idx, seg_localidx_to_pose, segment, history, dataset
# ):
#     pose_cur  = seg_localidx_to_pose[(segment, seg_idx)]
#     past_idxs = [max(seg_idx - i, 0) for i in range(history)]

#     preds_len = [len(seg_dataset[f]['pred_boxes']) for f in past_idxs]
#     max_props = max(preds_len) if max(preds_len) > 0 else 1
#     all_boxes, all_scores, all_labels = [], [], []

#     for frame_seg in past_idxs:
#         frame    = seg_dataset[frame_seg]
#         preds    = frame['pred_boxes']
#         scores   = frame['pred_scores']
#         labels   = frame['pred_labels']
#         pose_past = seg_localidx_to_pose[(segment, frame_seg)]

#         if isinstance(preds, torch.Tensor):
#             preds  = preds.detach().cpu().numpy()
#         else:
#             preds  = np.array(preds)
#         if isinstance(scores, torch.Tensor):
#             scores = scores.detach().cpu().numpy()
#         else:
#             scores = np.array(scores)

#         if isinstance(labels, torch.Tensor):
#             labels = labels.detach().cpu().numpy()
#         else:
#             labels = np.array(labels)

#         scores = scores.astype(np.float32)
#         labels = labels.astype(np.int64)

#         if preds.shape[0] > 0:
#             preds = preds.copy()
#             preds[:, 7:9] = -0.1 * preds[:, 7:9]
#             if frame_seg != seg_idx:
#                 preds = dataset.transform_prebox_to_current(preds, pose_past, pose_cur)

#         n = min(len(preds), max_props)
#         pad = max_props - n

#         box_arr   = np.zeros((max_props, preds.shape[1] if len(preds) > 0 else 9), dtype=np.float32)
#         score_arr = np.zeros(max_props, dtype=np.float32)
#         label_arr = np.zeros(max_props, dtype=np.float32)
#         if n > 0:
#             box_arr[:n]   = preds[:n]
#             score_arr[:n] = scores[:n]
#             label_arr[:n] = labels[:n]

#         all_boxes.append(box_arr)
#         all_scores.append(score_arr)
#         all_labels.append(label_arr)

#     proposals  = torch.tensor(np.stack(all_boxes,  0)[np.newaxis], dtype=torch.float32)  # (1,T,128,9)
#     roi_scores = torch.tensor(np.stack(all_scores, 0)[np.newaxis], dtype=torch.float32)  # (1,T,128)
#     roi_labels = torch.tensor(np.stack(all_labels, 0)[np.newaxis], dtype=torch.long)     # (1,T,128)
#     return proposals, roi_scores, roi_labels



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args, logger):
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'

    dataset, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    logger.info("Model loaded.")

    # Build index structures
    
    # global index --> (segment, local_index)
    global_to_segment = {}
    seg_frame_counters = {}
    
    #(segment, local_idx)--> pose
    seg_localidx_to_pose = {}
    for i, info in enumerate(dataset.infos):
        seg = info['point_cloud']['lidar_sequence']
        if seg not in seg_frame_counters:
            seg_frame_counters[seg] = 0
        local_idx = seg_frame_counters[seg]
        global_to_segment[i] = (seg, local_idx)
        seg_localidx_to_pose[(seg, local_idx)] = info['pose'].reshape((4, 4))
        seg_frame_counters[seg] += 1

    ordered_segments = list(dict.fromkeys(
        info['point_cloud']['lidar_sequence'] for info in dataset.infos
    ))
    
    # debugging shortcut to limit runtime
    if args.max_segments is not None:
        ordered_segments = ordered_segments[:args.max_segments]
        logger.info(f"Limited to {args.max_segments} segments")

    # Pre-compute last global index belonging to any ordered segment so we can
    # break the loop early instead of iterating all 38597 frames doing nothing.
    ordered_set = set(ordered_segments)
    max_ordered_global_idx = max(
        i for i, info in enumerate(dataset.infos)
        if info['point_cloud']['lidar_sequence'] in ordered_set
    )

    os.makedirs(args.out_dir, exist_ok=True)
    
    #chooses which annotation to use (PTT vs MSF)
    timing_key = f"timing_{args.detector}"

    #per segment states
    current_segment  = None
    seg_dataset      = None
    seg_timing_lookup = None
    hf               = None   # current open HDF5 file

    
    try:
        
        with PTTInstrumentationContext(
            model.roi_head, capture_attn=args.capture_attn
        ) as ctx:

            #iterate through frames
            for i_batch, batch in enumerate(tqdm(test_loader, desc="Inference")):

                # Stop once we've passed the last frame of any ordered segment
                if i_batch > max_ordered_global_idx:
                    break

                #extract segment names
                segment = batch['frame_id'][0].rsplit("_", 1)[0]
                if segment not in ordered_segments:
                    continue
                
                #get segment-relative index
                seg_idx = global_to_segment[i_batch][1]

                # --- segment transition ---
                if current_segment != segment:
                    
                    #close previous file
                    if hf is not None:
                        hf.close()
                        logger.info(f"Saved features for {current_segment}")

                    current_segment = segment
                    out_path = os.path.join(
                        args.out_dir, f"{segment}_features.h5"
                    )

                    if os.path.exists(out_path):
                        logger.info(f"Skipping {segment} (already done)")
                        hf = None
                        seg_dataset = None
                        continue

                    #open new HDF5 file
                    hf = h5py.File(out_path, 'w')
                    hf.attrs['segment']    = segment
                    hf.attrs['num_frames'] = seg_frame_counters.get(segment, 0)
                    
                    #open dataset for the segment
                    with open(
                        os.path.join(args.dataset, f"{segment}_d.pkl"), 'rb'
                    ) as f:
                        seg_dataset = pkl.load(f)

                    with open(
                        os.path.join(args.timing_annos, f"{segment}_a.pkl"), 'rb'
                    ) as f:
                        seg_annos = pkl.load(f)
                    timing_arr = np.array(seg_annos[timing_key])
                    seg_timing_lookup = build_timing_lookup(timing_arr)
                    logger.info(f"Processing {segment}")

                if hf is None:
                    continue   # skipped segment
                
                #get spoofed frame data
                frame_data = seg_dataset[seg_idx]
                assert batch['frame_id'][0] == frame_data['frame_id']

                #position in spoof timeline 
                k_value     = seg_timing_lookup.get(seg_idx, None)
                
                # Build proposal tensor and inject into batch
                # --- Build batch_mod via prepare_data (mirrors eval_ptt_final.py) ---
                past_frames_list = [max(seg_idx - i, 0) for i in range(args.history)]
                pose_cur = seg_localidx_to_pose[(segment, seg_idx)]

                past_boxes_np = []
                for frame_seg in past_frames_list:
                    preds = seg_dataset[frame_seg]['pred_boxes']
                    if isinstance(preds, torch.Tensor):
                        preds = preds.detach().cpu().numpy()
                    else:
                        preds = np.array(preds)
                    if preds.shape[0] > 0:
                        preds = preds.copy()
                        preds[:, 7:9] = -0.1 * preds[:, 7:9]
                        pose_pre = seg_localidx_to_pose.get((segment, frame_seg))
                        if pose_pre is not None and frame_seg != seg_idx:
                            preds = dataset.transform_prebox_to_current(preds, pose_pre, pose_cur)
                    past_boxes_np.append(preds)

                preds_len = [b.shape[0] for b in past_boxes_np]
                max_preds = max(preds_len) if max(preds_len) > 0 else 1

                past_boxes, past_scores, past_labels = [], [], []
                for frame_seg, preds_np in zip(past_frames_list, past_boxes_np):
                    scores = to_numpy(seg_dataset[frame_seg]['pred_scores']).astype(np.float32)
                    labels = to_numpy(seg_dataset[frame_seg]['pred_labels']).astype(np.float32)
                    pad_size = max_preds - preds_np.shape[0]
                    past_boxes.append( np.pad(preds_np, ((0, pad_size), (0, 0)), mode='constant'))
                    past_scores.append(np.pad(scores,   (0, pad_size),           mode='constant'))
                    past_labels.append(np.pad(labels,   (0, pad_size),           mode='constant'))

                past_boxes  = np.stack(past_boxes,  axis=0).astype(np.float32)
                past_scores = np.stack(past_scores, axis=0).astype(np.float32)
                past_labels = np.stack(past_labels, axis=0).astype(np.float32)

                dict_mod = dataset[i_batch].copy()
                dict_mod['roi_boxes']  = past_boxes
                dict_mod['roi_scores'] = past_scores
                dict_mod['roi_labels'] = past_labels
                dict_mod['points']     = seg_dataset[seg_idx]['points']

                dict_mod  = helpers_ptt.inject_gt_names(dict_mod, dataset.class_names)
                dict_mod  = dataset.prepare_data(dict_mod)
                batch_mod = dataset.collate_batch([dict_mod])
                load_data_to_gpu(batch_mod)

                # --- Forward pass with instrumentation ---
                with torch.no_grad():
                    model(batch_mod)

                
                #retrieve captured data (activations/features, attention matrices)
                acts, attn = ctx.flush()
                
                #ground truth extraction 
                real_gt  = frame_data.get('gt_boxes',  None)
                spoof_gt = frame_data.get('spoof_gt',  None)
                
                #build proposal level records
                records = collect_proposal_features(
                    acts=acts,
                    attn=attn,
                    batch_dict=batch_mod,
                    spoof_gt=spoof_gt,
                    real_gt_boxes=real_gt,
                    iou_fn=iou3d_nms_utils.boxes_iou3d_gpu,
                )

                # Drop background proposals that have no meaningful phantom overlap.
                # Keep: all real_tp / spoofed_fp / clean_fp, plus any background
                # proposal whose IoU with the phantom >= 0.7 (could be a suppressed
                # spoofed detection worth inspecting).
                records = [
                    r for r in records
                    if r['category'] != 'background' or r['iou_spoof'] >= 0.7
                ]

                #write to disk 
                write_frame_to_hdf5(
                    hf,
                    frame_id=batch_mod['frame_id'][0],
                    seg_idx=seg_idx,
                    k_value=k_value,
                    records=records
                )

    except Exception:
        if hf is not None:
            hf.close()
        raise

    if hf is not None:
        hf.close()
        logger.info(f"Saved features for {current_segment}")

    logger.info("Done.")


def main():
    args = parse_args()
    logger = setup_logger(args.log_file)
    try:
        run(args, logger)
    except Exception as e:
        logger.error("FAILED: " + str(e))
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
