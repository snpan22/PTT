import os
from pathlib import Path
import copy
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

import random

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


import torch.nn.functional as F

from collections import defaultdict
import pickle as pkl


if __name__ == '__main__':
    cfg.clear()


    CFG_FILE = 'tools/cfgs/waymo_models/ptt_32frames.yaml' 
    CKPT = 'output/cfgs/waymo_models/default/checkpoint_epoch_6.pth'  # <- put your ckpt here

    cfg_from_yaml_file(CFG_FILE, cfg)
    cfg.TAG = Path(CFG_FILE).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'

    # cfg.DATA_CONFIG.DATA_PATH = os.path.join(OPENPCDET_PATH, 'data', 'waymo')

    logger = common_utils.create_logger()
    logger.info(f'Loaded cfg from {CFG_FILE}')


    dataset, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=False
    )

    len_test = len(dataset)
    logger.info(f'Test set length: {len_test}')
    

    data_iter = iter(test_loader)
    batch_dict = next(data_iter)




    with open('asr/asr_nomsf.pkl', 'rb') as file:
        data = pkl.load(file)
        
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    ordered_segments = [info['point_cloud']['lidar_sequence'] for info in dataset.infos]
    ordered_segments = list(dict.fromkeys(ordered_segments))  # remove duplicates while preserving order

    pred_dir = 'preds_final_msf8'
    dataset_dir = 'datasets'

    dataset_names = os.listdir(dataset_dir)
    dataset_names = [name for name in dataset_names if 'closer' not in name and 'random' not in name and 'removal' not in name]
    pred_names = os.listdir(pred_dir)
    pred_names = [name for name in pred_names if 'removal' not in name]
    
    asr_annos = []
    for pred in pred_names:
        print(pred)
        
        mode = pred.split('_', 1)[1]
        asr_annos_det_mode = {}

        asr_annos_det_mode['name'] = pred
        segment_preds = {}
        
        current_segment = None
        seg_dataset = None
        
        total_target_frames = 0
        num_vehicle_fp = 0
        num_ped_fp = 0
        num_cyc_fp = 0
        num_0_preds = 0

        scores_fp_vehicles = []
        scores_fp_ped = []
        scores_fp_cyc = []
        spoof_rc_survivability_vehicles = []
        spoof_rc_survivability_ped = []
        spoof_rc_survivability_cyc = []

        segment_count = 0
        
        segment_preds = {}
                
        for segment_name in ordered_segments:
            with open(f"{pred_dir}/{pred}/{segment_name}_p.pkl", "rb") as f:
                segment_preds[segment_name] = pkl.load(f)
    

        for info in dataset.infos:
            frame_id = info['frame_id']
            parts = frame_id.rsplit('_', 1)
            segment = parts[0]
            i_seg = int(parts[1])

            if current_segment != segment:
                segment_count += 1
                # print(f"segment count = {segment_count}")

                seg_dataset_file = f"{dataset_dir}/{mode}_d/{segment}_d.pkl"
                # print(seg_dataset_file)
                with open(seg_dataset_file, "rb") as f:
                    seg_dataset = pkl.load(f)

                current_segment = segment

            assert frame_id == seg_dataset[i_seg]['frame_id']

            frame = seg_dataset[i_seg]
        
            frame_pred = segment_preds[segment][i_seg]
            
            gt_spoof = frame['spoof_gt']
            if gt_spoof is not None:
                total_target_frames += 1
                pred_boxes = frame_pred['boxes_lidar']

                if pred_boxes.shape[0] == 0:
                    num_0_preds += 1
                    # print("0 preds")
                    continue
                gt_boxes = frame['gt_boxes']

                scores = frame_pred['score']
                labels = frame_pred['pred_labels']

                pred = torch.tensor(pred_boxes[:, :7]).cuda().float()

                gt = torch.tensor(gt_spoof[:7]).unsqueeze(0).cuda().float()

                iou = iou3d_nms_utils.boxes_iou3d_gpu(pred, gt)
                max_iou = iou.max().item()
                idx = torch.argmax(iou).item()
                spoof_score = scores[idx]
                spoof_label = labels[idx]

                n_spoof_r = frame['n_spoof_r']
                n_spoof_k = frame['n_spoof_k']

                if(max_iou >= 0.7 and spoof_label == 1):
                    # print(frame_id)
                    # print(max_iou, spoof_score)
                    num_vehicle_fp += 1
                    scores_fp_vehicles.append(spoof_score)
                    spoof_rc_survivability_vehicles.append((n_spoof_r, n_spoof_k, n_spoof_k/n_spoof_r))
                if(spoof_label == 2 and max_iou >= 0.5):
                    num_ped_fp += 1
                    scores_fp_ped.append(spoof_score)
                    spoof_rc_survivability_ped.append((n_spoof_r, n_spoof_k, n_spoof_k/n_spoof_r))
                    # print("spoof misclasified as pedestrian")
                if(spoof_label == 3 and max_iou >= 0.5):
                    num_cyc_fp +=1
                    scores_fp_cyc.append(spoof_score)
                    spoof_rc_survivability_cyc.append((n_spoof_r, n_spoof_k, n_spoof_k/n_spoof_r))
                    # print("spoof misclassified as cyclist")
            

        asr_annos_det_mode['asr_vehicle'] = num_vehicle_fp/total_target_frames
        asr_annos_det_mode['asr_pedestrian'] = num_ped_fp/total_target_frames
        asr_annos_det_mode['asr_cyclist'] = num_cyc_fp/total_target_frames
        asr_annos_det_mode['zero_pred_rate'] = num_0_preds / total_target_frames
        asr_annos_det_mode['scores_vehicle'] = np.array([t.item() for t in scores_fp_vehicles])
        asr_annos_det_mode['scores_pedestrian'] = np.array([t.item() for t in scores_fp_ped])
        asr_annos_det_mode['scores_cyclist'] = np.array([t.item() for t in scores_fp_cyc])
        asr_annos_det_mode['spoof_surv_vehicle'] = spoof_rc_survivability_vehicles
        asr_annos_det_mode['spoof_surv_pedestrian'] = spoof_rc_survivability_ped
        asr_annos_det_mode['spoof_surv_cyclist'] = spoof_rc_survivability_cyc
        asr_annos_det_mode['num_targets'] = total_target_frames
        asr_annos.append(asr_annos_det_mode)
        print(f"{pred}: {asr_annos_det_mode['asr_vehicle']}")

    data.extend(asr_annos)
    with open(f"asr/asr_pi.pkl", "wb") as f:
        pkl.dump(asr_annos, f)
    
    