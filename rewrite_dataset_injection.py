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
importlib.reload(helpers_ptt)
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

    parser.add_argument("--cp_sf_cfg", required=True)
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to runtime log file"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--old_dataset_dir", required=True)
    parser.add_argument("--save_dir_dataset", required=True)

    # parser.add_argument("--save_dir", default="segment_ckpts")

    return parser.parse_args()





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

    # Build dict: frame_id -> index
    sf_dict = {info['frame_id']: i for i, info in enumerate(dataset_sf.infos)}
    mf_dict = {info['frame_id']: i for i, info in enumerate(dataset_mf.infos)}
    common_frame_ids = [fid for fid in sf_dict if fid in mf_dict]
    frame_ids = [info['frame_id'] for info in dataset_sf.infos]

    unique_segments_sf = sorted(list(set(fid.rsplit('_', 1)[0] for fid in frame_ids)))
        
    mf_segments = set(info['point_cloud']['lidar_sequence']
                    for info in dataset_mf.infos)
    common_segments = mf_segments.intersection(unique_segments_sf)

    # mapping: sf index -> mf index
    sf_to_mf = {sf_dict[fid]: mf_dict[fid] for fid in common_frame_ids}


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
            
            
    dataset_files = os.listdir(args.old_dataset_dir)

    ordered_segments = [info['point_cloud']['lidar_sequence'] for info in dataset_sf.infos]
    ordered_segments = list(dict.fromkeys(ordered_segments))  # remove duplicates while preserving order

    # map filenames to segment names
    file_map = {f.replace('_d.pkl',''): f for f in dataset_files}
    # reorder files
    dataset_files_ordered = [file_map[s] for s in ordered_segments if s in file_map]


    # seg_dataset_file = f"{args.dataset}/{dataset_files_ordered[0]}"
    # seg_dataset_file = f"/storage/scratch1/9/spanse30/PTT/datasets/global_hard_d/{dataset_files_ordered[0]}"
    output_dir = args.save_dir_dataset
    os.makedirs(output_dir, exist_ok=True)
    done_segments = {f.replace('.pkl', '') for f in os.listdir(output_dir) if f.endswith('.pkl')}
    log.info("Found %d already-saved segments, will skip them", len(done_segments))

    # Open the file in read-binary mode ('rb')
    # with open(seg_dataset_file, 'rb') as file:
    #     # Use pickle.load() to deserialize the data
    #     seg_dataset = pkl.load(file)
    current_segment = None
    segment_frames = []
    try:
        for(frame,batch)  in enumerate(test_loader_mf):
            # if(frame <=197): continue
            segment = batch['frame_id'][0].rsplit("_", 1)[0]
            if segment not in common_segments:
                continue
            if segment in done_segments:
                continue
            seg_idx = global_to_segment[frame][1]
            if current_segment != segment:

                # save previous segment if it exists
                if current_segment is not None:
                    output_path = f"{output_dir}/{current_segment}.pkl"
                    with open(output_path, "wb") as f:
                        pkl.dump(segment_frames, f)
                    # print(f"saved {current_segment}")
                    log.info("Saved preds for : %s", current_segment)
                current_segment = segment
                # print(current_segment)
                segment_frames = []
                # load spoof dataset
                seg_dataset_file = f"{args.old_dataset_dir}/{current_segment}_d.pkl"
                with open(seg_dataset_file, "rb") as f:
                    seg_dataset = pkl.load(f)        
                
            assert batch['frame_id'][0] == seg_dataset[seg_idx]['frame_id']
        
            dict_mf_mod = dataset_mf[frame].copy()
            for k in ['voxels', 'voxel_coords', 'voxel_num_points']:
                dict_mf_mod.pop(k, None)
            dict_mf_mod['points'] = seg_dataset[seg_idx]['points']
            dict_mf_mod['pose'] = seg_dataset[seg_idx]['pose']
            dict_mf_mod['spoof_gt'] = seg_dataset[seg_idx]['spoof_gt']
            
            segment_frames.append(dict_mf_mod)
    except Exception:
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        raise 
    if segment_frames:
        output_path = f"{output_dir}/{current_segment}.pkl"
        with open(output_path, "wb") as f:
            pkl.dump(segment_frames, f)
        log.info("Saved dataset for : %s", current_segment)
            
    return

if __name__ == "__main__":
    main()
