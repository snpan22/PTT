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
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_file', default="waymo_eval.log")
    parser.add_argument('--metrics_out', default="metrics.json")
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
        


    dataset_files = os.listdir(args.dataset)

    ordered_segments = [info['point_cloud']['lidar_sequence'] for info in dataset.infos]
    ordered_segments = list(dict.fromkeys(ordered_segments))  # remove duplicates while preserving order

    # map filenames to segment names
    file_map = {f.replace('_d.pkl',''): f for f in dataset_files}
    # reorder files
    dataset_files_ordered = [file_map[s] for s in ordered_segments if s in file_map]

    seg_dataset_file = f"{args.dataset}/{dataset_files_ordered[0]}"
    # Open the file in read-binary mode ('rb')
    with open(seg_dataset_file, 'rb') as file:
        # Use pickle.load() to deserialize the data
        seg_dataset = pkl.load(file)


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

                # load spoof dataset
                seg_dataset_file = f"{args.dataset}/{current_segment}_d.pkl"
                with open(seg_dataset_file, "rb") as f:
                    seg_dataset = pkl.load(f)
            
                    
            assert batch['frame_id'][0] == seg_dataset[seg_idx]['frame_id']

            
            past_frames_list = [max(seg_idx - i, 0) for i in range(history)]

            # Current frame pose for coordinate transforms
            pose_cur = seg_localidx_to_pose[(segment, seg_idx)]

            # Convert boxes from each historical frame into current frame coordinates,
            # matching dataset.get_sequence_data / transform_prebox_to_current.
            # Also apply velocity scaling: raw vx/vy -> -0.1 * vx/vy (backwards motion
            # vector), matching dataset.load_pred_boxes_from_dict line 268.
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

            past_boxes  = []
            past_scores = []
            past_labels = []

            for frame_seg, preds_np in zip(past_frames_list, past_boxes_np):
                scores = seg_dataset[frame_seg]['pred_scores']
                labels = seg_dataset[frame_seg]['pred_labels']

                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().numpy()

                scores = np.array(scores, dtype=np.float32)
                labels = np.array(labels, dtype=np.float32)

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
            dict_mod['points'] = seg_dataset[seg_idx]['points']
            
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