import torch
import torch.nn.functional as F
import argparse
import pickle as pkl
import logging
import json
from pathlib import Path
import traceback
import os
import sys
import numpy as np
from tqdm import tqdm
from pcdet.utils import common_utils

import importlib
import helpers_ptt
importlib.reload(helpers_ptt)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logger(log_path):

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_pkl', required=True,
                        help='Path to clean CenterPoint result_fixed.pkl')
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_file', default="ptt_sanity_manual.log")
    parser.add_argument('--metrics_out', default="metrics_ptt_sanity_manual.json")
    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_evaluations(args, logger):

    # args = parse_args()
    # logger = setup_logger(args.log_file)

    logger.info("Loading config...")
    cfg_from_yaml_file(args.cfg_file, cfg)

    logger.info("Building dataset (GT only, no model)...")
    
    
    CFG_FILE = args.cfg_file
    CKPT = args.ckpt

    cfg_from_yaml_file(CFG_FILE, cfg)
    cfg.TAG = Path(CFG_FILE).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'

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
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )

    logger.info(f'Loading checkpoint from: {CKPT}')
    model.load_params_from_file(filename=CKPT, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    
    
    global_to_segment = {}
    segment_frame_counters = {}
    # (segment, local_idx) -> pose (4,4) for coordinate frame transforms
    seg_localidx_to_pose = {}
    for i, info in enumerate(dataset.infos):
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        local_idx = segment_frame_counters[seg]
        global_to_segment[i] = (seg, local_idx)
        seg_localidx_to_pose[(seg, local_idx)] = info['pose'].reshape((4, 4))
        segment_frame_counters[seg] += 1
        


    ordered_segments = [info['point_cloud']['lidar_sequence'] for info in dataset.infos]
    ordered_segments = list(dict.fromkeys(ordered_segments))

    # Load clean CenterPoint proposals and index by (seg, local_idx)
    logger.info(f'Loading clean proposals from: {args.result_pkl}')
    with open(args.result_pkl, 'rb') as f:
        clean_result = pkl.load(f)

    # Build (seg, local_idx) -> pred_boxes (N,9) lookup
    # result_fixed.pkl boxes are (N,9): [x,y,z,dx,dy,dz,heading,vx,vy]
    # score and label are separate fields; we concatenate to match the
    # (N,11) format that load_pred_boxes_from_dict expects before slicing
    clean_proposals = {}
    result_frame_counters = {}
    for frame_dict in clean_result:
        fid = frame_dict['frame_id']
        seg = fid.rsplit('_', 1)[0]
        if seg not in result_frame_counters:
            result_frame_counters[seg] = 0
        loc_idx = result_frame_counters[seg]
        result_frame_counters[seg] += 1

        boxes  = frame_dict['pred_boxes']   # (N,9) numpy
        scores = frame_dict['pred_scores']  # (N,)
        labels = frame_dict['pred_labels']  # (N,)
        if isinstance(boxes, torch.Tensor):
            boxes  = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
        # stack to (N,11) so downstream slicing [:, 0:9] works cleanly
        boxes_11 = np.concatenate([
            boxes,
            scores[:, np.newaxis].astype(np.float32),
            labels[:, np.newaxis].astype(np.float32)
        ], axis=-1)
        clean_proposals[(seg, loc_idx)] = boxes_11

    logger.info(f'Clean proposals loaded for {len(result_frame_counters)} segments')


    save_dir_preds = args.pred_dir
    os.makedirs(save_dir_preds, exist_ok=True)

    completed = set(
        f[:-4] for f in os.listdir(save_dir_preds)
    )

    history = 32
    segment_preds_list = []
    current_segment = None
    try: 
        for (i_ptt,batch)  in enumerate(test_loader):
            
            segment = batch['frame_id'][0].rsplit("_", 1)[0]
            seg_idx = global_to_segment[i_ptt][1]

            # skip completed segments immediately
            if f"{segment}_p" in completed:
                continue

            # initialize or switch segment
            if current_segment != segment:

                # save previous segment if it exists
                if current_segment is not None:
                    pred_path = f"{save_dir_preds}/{current_segment}_p.pkl"
                    with open(pred_path, "wb") as f:
                        pkl.dump(segment_preds_list, f)
                    logger.info("Saved preds for : %s", current_segment)

                segment_preds_list = []
                current_segment = segment

            
            past_frames_list = [max(seg_idx - i, 0) for i in range(history)]

            # Current frame pose for coordinate transforms
            pose_cur = seg_localidx_to_pose[(segment, seg_idx)]

            # Convert boxes from each historical frame into current frame coordinates,
            # matching dataset.get_sequence_data / transform_prebox_to_current.
            # Also apply velocity scaling: raw vx/vy -> -0.1 * vx/vy (backwards motion
            # vector), matching dataset.load_pred_boxes_from_dict line 268.
            past_boxes_np = []
            for frame_seg in past_frames_list:
                boxes_11 = clean_proposals.get((segment, frame_seg))
                if boxes_11 is None:
                    preds = np.zeros((0, 9), dtype=np.float32)
                else:
                    preds = boxes_11[:, :9].copy()

                if preds.shape[0] > 0:
                    preds[:, 7:9] = -0.1 * preds[:, 7:9]
                    pose_pre = seg_localidx_to_pose.get((segment, frame_seg))
                    if pose_pre is not None and frame_seg != seg_idx:
                        preds = dataset.transform_prebox_to_current(preds, pose_pre, pose_cur)

                past_boxes_np.append(preds)

            preds_len = [b.shape[0] for b in past_boxes_np]
            max_preds = max(preds_len) if max(preds_len) > 0 else 1

            past_boxes  = []
            past_scores = []
            past_labels = []

            for frame_seg, preds_np in zip(past_frames_list, past_boxes_np):
                boxes_11 = clean_proposals.get((segment, frame_seg))
                if boxes_11 is not None and boxes_11.shape[0] > 0:
                    scores = boxes_11[:, 9].astype(np.float32)
                    labels = boxes_11[:, 10].astype(np.float32)
                else:
                    scores = np.zeros(0, dtype=np.float32)
                    labels = np.zeros(0, dtype=np.float32)

                pad_size = max_preds - preds_np.shape[0]
                past_boxes.append( np.pad(preds_np, ((0, pad_size), (0, 0)), mode='constant'))
                past_scores.append(np.pad(scores,   (0, pad_size),           mode='constant'))
                past_labels.append(np.pad(labels,   (0, pad_size),           mode='constant'))

            past_boxes  = np.stack(past_boxes,  axis=0).astype(np.float32)
            past_scores = np.stack(past_scores, axis=0).astype(np.float32)
            past_labels = np.stack(past_labels, axis=0).astype(np.float32)
            
            dict_mod = dataset[i_ptt].copy()
            dict_mod['roi_boxes'] = past_boxes
            dict_mod['roi_scores'] = past_scores
            dict_mod['roi_labels'] = past_labels
            # Use clean points from dataset directly (ONLY_CURRENT=True)
            
            dict_mod = helpers_ptt.inject_gt_names(dict_mod, dataset.class_names)
            dict_mod = dataset.prepare_data(dict_mod)
            batch_mod = dataset.collate_batch([dict_mod])
            
            load_data_to_gpu(batch_mod)

            with torch.no_grad():
                pred_dicts, _ = model(batch_mod)

            # print(f"================================MAKING PTT PREDICTION {seg_idx} ===============================")
            annos = dataset.generate_prediction_dicts(
                batch_mod,
                pred_dicts,
                cfg.CLASS_NAMES
            )

            segment_preds_list+=annos
            
            
    except Exception:
        logger.error("===== EXCEPTION =====")
        logger.error(traceback.format_exc())
        raise  
        
    finally:
        if current_segment is not None and len(segment_preds_list) > 0:
            pred_path = f"{save_dir_preds}/{current_segment}_p.pkl"
            with open(pred_path, "wb") as f:
                pkl.dump(segment_preds_list, f)
            logger.info("Saved preds for : %s", current_segment)

    
    

    pred_dir = args.pred_dir

    segment_preds = {}
    unique_segment_names = dataset.seq_name_to_infos.keys()
    for segment_name in unique_segment_names:
        with open(f"{pred_dir}/{segment_name}_p.pkl", "rb") as f:
            segment_preds[segment_name] = pkl.load(f)
    infos = dataset.infos

    det_annos = []

    segment_frame_counters = {}

    for info in infos:
        segment = info['point_cloud']['lidar_sequence']
        
        if segment not in segment_frame_counters:
            segment_frame_counters[segment] = 0
        
        idx = segment_frame_counters[segment]
        
        det_annos.append(segment_preds[segment][idx])
        
        segment_frame_counters[segment] += 1
    for pred, info in zip(det_annos, dataset.infos):
        assert pred['frame_id'] == info['frame_id']
    # --------------------------------------------------------
    # Progress indicator during evaluation
    # --------------------------------------------------------
    # dataset.evaluation() is monolithic, so we simulate progress
    # by wrapping the call — still useful for tracking runtime
    # --------------------------------------------------------

    logger.info("\n\nStarting Waymo evaluation")

    for _ in tqdm(range(1), desc="Waymo Metrics"):
        result_str, result_dict = dataset.evaluation(
            det_annos,
            cfg.CLASS_NAMES,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

    logger.info("Evaluation complete")

    logger.info("\n===== RESULT STRING =====\n")
    logger.info("\n" + result_str)

    logger.info("\n===== RESULT DICT =====")
    logger.info(json.dumps(result_dict, indent=2, default = float))

    # Save metrics JSON
    with open(args.metrics_out, "w") as f:
        json.dump(result_dict, f, indent=2, default = float)

    logger.info(f"Metrics saved to {args.metrics_out}")

def main():

    args = parse_args()
    logger = setup_logger(args.log_file)

    try:
        run_evaluations(args, logger)

    except Exception as e:
        logger.error("===== EVALUATION FAILED =====")
        logger.error(str(e))
        logger.error("\nFull traceback:\n")
        logger.error(traceback.format_exc())

        # Also print to stderr so Slurm captures it
        print(traceback.format_exc(), file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()