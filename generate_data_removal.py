#!/usr/bin/env python
import os
import sys
import argparse
import importlib
import pickle as pkl
import logging
import json
from pathlib import Path
import traceback
from tqdm import tqdm
import time 
import torch
import numpy as np
import random

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import helpers_ptt
import removal
importlib.reload(helpers_ptt)
importlib.reload(removal)
import torch.nn.functional as F

from collections import defaultdict

def unload_pcdet():
    for k in list(sys.modules.keys()):
        if k.startswith("pcdet"):
            del sys.modules[k]


def import_pcdet(repo_path, alias):
    """
    Import a specific OpenPCDet repo under a unique namespace.
    """
    sys.path.insert(0, repo_path)
    module = importlib.import_module("pcdet")
    sys.modules[alias] = module
    sys.path.pop(0)
    return module

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cp_mf_root", required=True)

    parser.add_argument("--cp_mf_cfg", required=True)
    parser.add_argument("--cp_mf_ckpt", required=True)

    parser.add_argument("--cp_sf_cfg", required=True)
    parser.add_argument("--cp_sf_ckpt", required=True)
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to runtime log file"
    )
    parser.add_argument("--save_dir_dataset", required=True)
    parser.add_argument("--save_dir_annos", required=True)
    parser.add_argument("--az_width", type=int, required=True,
                    help="Wedge angle in degrees")

    # parser.add_argument("--save_dir", default="segment_ckpts")

    return parser.parse_args()


def process_4frames(dataset_mf,
                    dataset_sf, 
                    frame_idx, 
                    target_frames, 
                    frame_seq, 
                    removal_cache,
                    mf_to_sf_global):
    
    #frame idx will be an index into centerpoint dataset
    # print(f"4frame point count before spoofing: {dataset_mf[frame_idx]['points'].shape}")
    points_frame = None
    #put global spoof in ego coordinates of lag = 0 frame
    none_spoofed = 1
    infos = dataset_mf.infos
    info_cur = infos[frame_idx]
    seq_name = info_cur['point_cloud']['lidar_sequence']
    pose_cur = info_cur['pose'].reshape(4, 4) #ego pose of current frame
    # print(seq_name)

    #acquire list of frames in set of 4 (centerpoint multiframe) that need to be inspected for modification
    indices = []
    for i in range(4):
        target_idx = frame_idx - i

        if target_idx < 0:
            target_idx = 0

        if infos[target_idx]['point_cloud']['lidar_sequence'] != seq_name:
            target_idx = indices[-1] if indices else frame_idx

        indices.append(target_idx)
            
    # print(f"cp 4 frames indices processed: {indices}")
    has_spoof= any(target_frames[i] is not None for i in indices)
    #if none of the frames are marked for spoofing, just return the original points for the current frame (no raycasting needed)
    if not has_spoof:
        # print("not spoofing")
        pts = dataset_sf[mf_to_sf_global[frame_idx]]['points']
        points_frame = np.hstack([pts, np.zeros((pts.shape[0], 1))])
        return dataset_mf[frame_idx]['points'], none_spoofed, points_frame

    #iterate through 4 frames 
    merged_points_list = []
    for i, idx in enumerate(indices):
     
        
        #get points from single frame, add lag 0
        pts = dataset_sf[mf_to_sf_global[idx]]['points'].copy()
        points= np.hstack([pts, np.zeros((pts.shape[0], 1))])
        
        #update lag 0 if needed
        points[:, -1] = (frame_idx - idx)*0.1
        pose_lag_i_frame = infos[idx]['pose'].reshape(4, 4)

        

        if(target_frames[idx] is not None):            
            ##???????????
            
            #use these for building the dataset
            # if idx == frame_idx, then this means this is lag 0, and it is already in lag 0 coordinates
            frame_points_spoof = removal_cache[idx].copy()
            frame_points_spoof[:, -1] = (frame_idx - idx) * 0.1  # set correct lag
            if(idx == frame_idx):
                points_frame = frame_points_spoof
                
                                                           
            # convert these back to lag 0 coordintaes 
            frame_points_spoof_hom = np.hstack((frame_points_spoof[:, :3], np.ones(frame_points_spoof.shape[0]).reshape(-1, 1)))
            frame_points_spoof_lag0_coords = frame_points_spoof_hom @ pose_lag_i_frame.T @ np.linalg.inv(pose_cur).T
                                                
            
            # print(f"spoofing frame {idx}, points diff = {frame_points_spoof_rc_lag0_coords.shape[0] - points.shape[0]}")
            points = np.hstack((frame_points_spoof_lag0_coords[:, :-1], frame_points_spoof[:, 3:]))
        else:
            # not spoofing, must convert back to lag 0 coordinates
            if(idx == frame_idx):
                points_frame = points
                
            points_last3cols = points[:, 3:]
            frame_points_hom = np.hstack((points[:, :3], np.ones(points.shape[0]).reshape(-1, 1)))
            frame_points_lag0_coords = frame_points_hom @ pose_lag_i_frame.T @ np.linalg.inv(pose_cur).T
            points = np.hstack((frame_points_lag0_coords[:, :-1], points_last3cols))
         
       
        # print(points.shape)
        merged_points_list.append(points)
    merged_points_array = np.vstack(merged_points_list)
    merged_points_array = merged_points_array.astype(np.float32)
    # print(f"4frame point count after spoofing: {merged_points_array.shape}")    

    
    none_spoofed = 0
    return merged_points_array, none_spoofed, points_frame


def track_sequence(dataset, start=32, end=63, desired_range=30.0,
                   target_class=1, max_step=6.0, negative = False):
    """
    Track a single GT object across frames.

    Parameters
    ----------
    dataset : indexable sequence where dataset[i] has keys
              'gt_boxes' (N,7+) and optionally 'gt_labels' (N,)
    start, end : int — frame range (inclusive)
    desired_range : float — preferred initial target range
    target_class : int — object class to track (1=vehicle)
    max_step : float — matching gate in meters

    Returns
    -------
    tracked_boxes : list of (box or None) per frame
    centers : (T, 3) array with NaN for lost frames
    diagnostics : dict
    """
    T = end - start + 1
    tracked_boxes = [None] * T
    tracked_idxs = [None] * T
    centers = np.full((T, 3), np.nan, dtype=float)
    match_dists = np.full(T, np.nan)
    ranges = np.full(T, np.nan)

    # --- Init ---
    init_data = dataset[start]
    init_boxes = init_data['gt_boxes']

    init_idx, cur_box = removal.select_initial_target(
        init_boxes,
        desired_range=desired_range, target_class=target_class, negative=negative
    )
    if cur_box is None:
        return tracked_boxes, centers, {"ok": False, "reason": "no_init_target"}

    tracked_boxes[0] = cur_box
    tracked_idxs[0] = init_idx
    centers[0] = cur_box[:3]
    ranges[0] = removal._box_range(cur_box)

    prev_box = cur_box
    prev_prev_box = None

    # --- Track ---
    for t in range(1, T):
        frame_data = dataset[start + t]
        curr_boxes = frame_data['gt_boxes']

        new_box, new_idx, d = removal.match_with_velocity(
            prev_box, prev_prev_box, curr_boxes,
            target_class=target_class,
            max_step=max_step
        )

        if new_box is None:
            # Lost — stop tracking
            break

        tracked_boxes[t] = new_box
        tracked_idxs[t] = new_idx
        centers[t] = new_box[:3]
        match_dists[t] = d
        ranges[t] = removal._box_range(new_box)

        prev_prev_box = prev_box
        prev_box = new_box

    # --- Diagnostics ---
    valid = ~np.isnan(centers[:, 0])
    valid_centers = centers[valid]
    num_tracked = int(np.sum(valid))
    if len(valid_centers) > 1:
        deltas = np.linalg.norm(np.diff(valid_centers, axis=0), axis=1)
    else:
        deltas = np.array([])
    # speeds_mps = deltas * 10.0  # approximate if 10 Hz
    
    valid_boxes = [b for b in tracked_boxes if b is not None]
    speeds_mps = np.array([np.linalg.norm([b[7], b[8]]) for b in valid_boxes])

    diagnostics = {
        "ok": True,
        "num_tracked": num_tracked,
        "total_frames": T,
        "init_range_m": float(ranges[0]),
        "range_trend": ranges[valid],
        "match_dists": match_dists,
        "step_deltas_m": deltas,
        "approx_speeds_mps": speeds_mps,
        "max_step_m": float(np.nanmax(deltas)) if len(deltas) > 0 else 0.0,        "max_speed_mps": float(np.nanmax(speeds_mps)) if num_tracked > 1 else 0.0,
        "mean_match_dist_m": float(np.nanmean(match_dists)) if num_tracked > 1 else 0.0,
    }
    
    return tracked_boxes, centers, diagnostics


def generate_timing_classification(target_frames, window_size):
    classification = []

    keys = sorted(target_frames.keys())
    n_prev = window_size - 1

    for k in keys:
        cur = target_frames.get(k)

        prev_vals = [target_frames.get(k - i) for i in range(1, n_prev + 1)]

        prev_all_none = all(v is None for v in prev_vals)
        prev_any_not_none = any(v is not None for v in prev_vals)

        if cur is None and prev_all_none:
            classification.append(0)

        elif cur is not None and prev_all_none:
            classification.append(1)

        elif cur is not None and prev_any_not_none:
            classification.append(2)

        elif cur is None and prev_any_not_none:
            classification.append(3)

    return classification
def extract_one_runs(d):
    runs = []
    current_run = []

    for idx in sorted(d.keys()):
        if d[idx] == 1:
            current_run.append(idx)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []

    # handle tail case
    if current_run:
        runs.append(current_run)

    return runs


def main():
    
    args = parse_args()
    log = logging.getLogger("spoof")
    log.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # Console handler 
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    log.addHandler(sh)

    # File handler 
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    
    
    #load in centerpoint multiframe dataset. will need this to modify rpn proposals
    unload_pcdet()
    sys.path.insert(0, args.cp_mf_root)

    from pcdet.config import cfg as cfg_cp
    from pcdet.config import cfg_from_yaml_file as cfg_from_yaml_file_cp
    from pcdet.datasets import build_dataloader as build_cp_loader
    from pcdet.models import build_network as build_cp_network
    from pcdet.models import load_data_to_gpu
    from pcdet.utils import common_utils

    sys.path.pop(0)

    # cfg_cp.TAG = 'centerpoint_multiframe'
    # cfg_cp.EXP_GROUP_PATH = 'waymo_multiframe'
    cfg_cp.clear()
    cfg_from_yaml_file_cp(args.cp_mf_cfg, cfg_cp)

    logger = common_utils.create_logger()
    logger.info(f'Loaded cfg from {args.cp_mf_cfg}')
    dataset_mf, test_loader_mf, _ = build_cp_loader(
        dataset_cfg=cfg_cp.DATA_CONFIG,
        class_names=cfg_cp.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=8,
        logger = logger,
        training=False
    )

    cp_mf_model = build_cp_network(
        model_cfg=cfg_cp.MODEL,
        num_class=len(cfg_cp.CLASS_NAMES),
        dataset=dataset_mf
    )
    cp_mf_model.load_params_from_file(args.cp_mf_ckpt,  logger=logger, to_cpu=False)
    cp_mf_model.cuda().eval()
    
    
    cfg_cp.clear()
    
    cfg_from_yaml_file_cp(args.cp_sf_cfg, cfg_cp)

    logger = common_utils.create_logger()
    logger.info(f'Loaded cfg from {args.cp_sf_cfg}')
    dataset_sf, _, _ = build_cp_loader(
        dataset_cfg=cfg_cp.DATA_CONFIG,
        class_names=cfg_cp.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=8,
        logger = logger,
        training=False
    )

    cp_sf_model = build_cp_network(
        model_cfg=cfg_cp.MODEL,
        num_class=len(cfg_cp.CLASS_NAMES),
        dataset=dataset_sf
    )
    cp_sf_model.load_params_from_file(args.cp_sf_ckpt,  logger=logger, to_cpu=False)
    cp_sf_model.cuda().eval()
    
    frame_ids = [info['frame_id'] for info in dataset_sf.infos]

    # Split the frame_id to get only the segment name (everything before the last underscore)
    # and use set() to find unique values
    unique_segments_sf = sorted(list(set(fid.rsplit('_', 1)[0] for fid in frame_ids)))
    
    
    
    mf_segments = set(info['point_cloud']['lidar_sequence']
                    for info in dataset_mf.infos)
    common_segments = mf_segments.intersection(unique_segments_sf)

    # Mapping: segment -> list of global indices (ordered)
    segment_to_global = defaultdict(list)

    # Mapping: global_idx -> (segment, local_idx)
    global_to_segment = {}

    for global_idx, info in enumerate(dataset_mf.infos):
        segment = info['point_cloud']['lidar_sequence']
        if segment in common_segments:
            local_idx = len(segment_to_global[segment])  # current position in segment
            segment_to_global[segment].append(global_idx)

            global_to_segment[global_idx] = (segment, local_idx)


    segment_counts = defaultdict(int)

    for info in dataset_mf.infos:
        segment = info['point_cloud']['lidar_sequence']
        if segment in common_segments: segment_counts[segment] += 1

    segment_last_local_idx = {seg: count - 1 for seg, count in segment_counts.items()}


    target_frames_global = {}
    target_frames_seg = {}

    for seg, last_ind in segment_last_local_idx.items():
        toggle = False
        targets_global = {}
        targets_seg = {}
        for i in range(last_ind+1):
            if i==0:
                targets_seg[i] = None
                targets_global[segment_to_global[seg][i]] = None
            else:
                rem = i%32
                # print(rem)
                if(not rem):
                    toggle = not toggle
                if toggle:
                    targets_seg[i] = 1
                    targets_global[segment_to_global[seg][i]] = 1
                else:
                    targets_seg[i] = None
                    targets_global[segment_to_global[seg][i]] = None
        target_frames_global[seg] = targets_global
        target_frames_seg[seg] = targets_seg


    sf_segment_to_global = defaultdict(list)

    for global_idx, info in enumerate(dataset_sf.infos):
        seg = info['frame_id'].rsplit('_', 1)[0]
        if seg in common_segments:
            sf_segment_to_global[seg].append(global_idx)
    mf_to_sf_global = {}

    for mf_global, (seg, local_idx) in global_to_segment.items():
        if seg in sf_segment_to_global and local_idx < len(sf_segment_to_global[seg]):
            sf_global = sf_segment_to_global[seg][local_idx]
            mf_to_sf_global[mf_global] = sf_global
            
    segment_dataset = []
    save_dir_dataset = args.save_dir_dataset
    os.makedirs(save_dir_dataset, exist_ok=True)
    save_dir_annos = args.save_dir_annos
    os.makedirs(save_dir_annos, exist_ok=True)
    
    completed = set(
        f[:-4] for f in os.listdir(save_dir_dataset)
    )
    rng = np.random.default_rng()
    target_frames = {}
    current_segment = None
    target_frames_anno = {}
    # prev_seq_name = None
    prr = {10:0.97,
       20:1,
       30:0.99,
       40:0.98,
       50:0.9,
       60:0.9
    }
    try:
        for (frame,batch)  in enumerate(test_loader_mf):
            # if(frame <=230): continue
            segment = batch['frame_id'][0].rsplit("_", 1)[0]
            if segment not in common_segments:
                continue

            if segment in completed:
                continue

            if current_segment is None:
                print(segment)
                current_segment = segment
                target_frames = target_frames_global[current_segment].copy()
                target_frames_anno = {k: None for k in target_frames_seg[current_segment]}


                #get spoofs for current segment
                # print(target_frames)
                runs_global = extract_one_runs(target_frames)            
                for run in runs_global:
                    subset_g = {k: target_frames[k] for k in run}
                    # subset_g_anno = {k: target_frames_anno[k] for k in run}
                    window_min = min(subset_g)
                    window_max = max(subset_g)

                    tracked_boxes, centers, diag = track_sequence(
                        dataset_mf, start=window_min, end=window_max, desired_range=60.0, target_class=1, negative = True
                    )
                    ##debug####
                    # x_center = [center[0] for center in centers]

                    # mask = ~np.isnan(centers).any(axis=1)

                    # # 2. Count them
                    # num_valid_rows = mask.sum()
                    # print(window_min, window_max, max(x_center), min(x_center), num_valid_rows)

                    ##end debug ##
                    for key, center, box in zip(subset_g.keys(), centers, tracked_boxes):
                        subset_g[key] = box


                    target_frames.update(subset_g)
                removal_cache = {}
                for run in runs_global:
                    state = None
                    for global_idx in sorted(run):
                        box = target_frames[global_idx]
                        if box is None:
                            target_frames_anno[global_to_segment[global_idx][1]] = None
                            target_frames[global_idx] = None
                            continue
                        pts = dataset_sf[mf_to_sf_global[global_idx]]['points'].copy()
                        points = np.hstack([pts, np.zeros((pts.shape[0], 1))])
                        az = args.az_width
                        removal_rate = prr[az]
                        attacked_points, state, meta = removal.spoof_frame_ahfr_roadside(
                            points=points,
                            target_box=box,       # aim using the tracked box
                            state=state,          # carries forward temporally
                            p_remove=removal_rate,
                            rng=rng,
                            az_width_deg = az
                        )
                        removal_cache[global_idx] = attacked_points
                        target_frames_anno[global_to_segment[global_idx][1]] = meta


            if(segment != current_segment):
                # temp = target_frames_seg[current_segment]
                # target_frames_anno_seg = {k: (target_frames_anno.get(k) if v is not None else None) for k, v in temp.items()}
                segment_annos = {}
                segment_annos['segment'] = current_segment
                segment_annos['timing_ptt'] = generate_timing_classification(target_frames_anno, 32)
                segment_annos['timing_msf_4'] = generate_timing_classification(target_frames_anno, 4)
                segment_annos['timing_msf_8'] = generate_timing_classification(target_frames_anno, 8)
                segment_annos['spoof_annos'] =  target_frames_anno

                dataset_path = f"{save_dir_dataset}/{current_segment}_d.pkl"
                anno_path = f"{save_dir_annos}/{current_segment}_a.pkl"

                with open(dataset_path,"wb") as f:
                    pkl.dump(segment_dataset,f)
                with open(anno_path,"wb") as f:
                    pkl.dump(segment_annos,f)

                log.info("Saved: %s", current_segment)

                segment_dataset = [] 
                current_segment = segment

                target_frames = {}
                target_frames_anno = {}
                current_segment = segment
                target_frames = target_frames_global[current_segment].copy()
                target_frames_anno = {k: None for k in target_frames_seg[current_segment]}
                runs_global = extract_one_runs(target_frames)            
                for run in runs_global:
                    subset_g = {k: target_frames[k] for k in run}
                    window_min = min(subset_g)
                    window_max = max(subset_g)

                    tracked_boxes, centers, diag = track_sequence(
                        dataset_mf, start=window_min, end=window_max, desired_range=60.0, target_class=1, negative = True
                    )
                    ##debug####
                    # x_center = [center[0] for center in centers]

                    # mask = ~np.isnan(centers).any(axis=1)

                    # # 2. Count them
                    # num_valid_rows = mask.sum()
                    # print(window_min, window_max, max(x_center), min(x_center), num_valid_rows)

                    ##end debug ##

                    for key, center, box in zip(subset_g.keys(), centers, tracked_boxes):
                        subset_g[key] = box


                    target_frames.update(subset_g)
                removal_cache = {}
                for run in runs_global:
                    state = None
                    for global_idx in sorted(run):
                        box = target_frames[global_idx]
                        if box is None:
                            target_frames_anno[global_to_segment[global_idx][1]] = None
                            target_frames[global_idx] = None

                            continue
                        pts = dataset_sf[mf_to_sf_global[global_idx]]['points'].copy()
                        points = np.hstack([pts, np.zeros((pts.shape[0], 1))])
                        az = args.az_width
                        removal_rate = prr[az]
                        attacked_points, state, meta = removal.spoof_frame_ahfr_roadside(
                            points=points,
                            target_box=box,       # aim using the tracked box
                            state=state,          # carries forward temporally
                            p_remove=removal_rate,
                            rng=rng,
                            az_width_deg = az
                        )
                        removal_cache[global_idx] = attacked_points
                        target_frames_anno[global_to_segment[global_idx][1]] = meta


            frame_data = {}
            mf_item = dataset_mf[frame].copy()
            frame_data['frame_id'] = mf_item['frame_id']
            frame_data['pose'] = mf_item['poses'][:4]
            frame_data['gt_boxes'] = mf_item['gt_boxes']

            frame_seq = global_to_segment[frame][1]
            four_frame_pts, none_spoofed, single_frame_points = process_4frames(dataset_mf,
                                                                                                dataset_sf, 
                                                                                                frame, 
                                                                                                target_frames, 
                                                                                                frame_seq, 
                                                                                                removal_cache,
                                                                                                mf_to_sf_global)
            if(none_spoofed):
                        # print("clean predictions")
                load_data_to_gpu(batch)
                cp_mf_model.eval()
                with torch.no_grad():
                    pred, _   = cp_mf_model(batch)
            else:
                # print("spoofed predictions")
                dict_mf_mod = dataset_mf[frame].copy()
                dict_mf_mod['batch_size']   = 1

                dict_mf_mod['points'] = four_frame_pts
                    #! rerun v oxelization and forward pass
                for k in ['voxels', 'voxel_coords', 'voxel_num_points']:
                    dict_mf_mod.pop(k, None)
                dict_mf_mod   = helpers_ptt.inject_gt_names(dict_mf_mod, dataset_mf.class_names)
                dict_mf_mod   = dataset_mf.prepare_data(dict_mf_mod)

                load_data_to_gpu(dict_mf_mod)
                batch_mf_mod = helpers_ptt.convert_to_batch_cp_mf(dict_mf_mod)

                cp_mf_model.eval()
                with torch.no_grad():
                    pred, _   = cp_mf_model(batch_mf_mod)

            frame_data['pred_boxes'] = pred[0]['pred_boxes']
            frame_data['pred_scores'] = pred[0]['pred_scores']
            frame_data['pred_labels'] = pred[0]['pred_labels']
            frame_data['points'] = single_frame_points
            frame_data['spoof_gt'] = target_frames[frame]
            
            segment_dataset.append(frame_data)

            
    except Exception:
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        raise  

    if segment_dataset:
        segment_annos = {}
        segment_annos['segment'] = current_segment
        segment_annos['timing_ptt'] = generate_timing_classification(target_frames_anno, 32)
        segment_annos['timing_msf_4'] = generate_timing_classification(target_frames_anno, 4)
        segment_annos['timing_msf_8'] = generate_timing_classification(target_frames_anno, 8)

        
        segment_annos['spoof_annos'] =  target_frames_anno

        dataset_path = f"{save_dir_dataset}/{current_segment}_d.pkl"
        anno_path = f"{save_dir_annos}/{current_segment}_a.pkl"

        with open(dataset_path,"wb") as f:
            pkl.dump(segment_dataset,f)
        with open(anno_path,"wb") as f:
            pkl.dump(segment_annos,f)

        log.info("Saved: %s", current_segment)
        
    return
    
if __name__ == "__main__":
    main()