import argparse
import pickle as pkl
import logging
import torch
from pathlib import Path
import traceback
import os
import sys
from tqdm import tqdm
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logger(log_path):
    logger = logging.getLogger("eval_history_curve")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file',     required=True)
    parser.add_argument('--pred_dir',     required=True,
                        help='Directory containing <segment>_p.pkl prediction files')
    parser.add_argument('--timing_annos', required=True,
                        help='Directory containing <segment>_a.pkl annotation files')
    parser.add_argument('--detector',     required=True, choices=['ptt', 'msf'],
                        help='Which timing key to read: timing_ptt or timing_msf')
    parser.add_argument('--dataset',      required=True,
                        help='Directory containing <segment>_d.pkl spoofed dataset files')
    parser.add_argument('--metrics_out',  default='history_curve.pkl')
    parser.add_argument('--log_file',     default='eval_history_curve.log')
    parser.add_argument('--workers',      type=int, default=4)
    return parser.parse_args()


# ------------------------------------------------------------
# Helper: find spoof block starts from timing array
# timing==1 marks the first frame of every spoofed block.
# ------------------------------------------------------------
def find_block_starts(timing_arr):
    return [i for i, t in enumerate(timing_arr) if t == 1]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_evaluations(args, logger):

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'
    logger.info(f'Loaded cfg from {args.cfg_file}')

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )
    logger.info(f'Test set length: {len(dataset)}')

    ordered_segments = list(dict.fromkeys(
        info['point_cloud']['lidar_sequence'] for info in dataset.infos
    ))

    # ----------------------------------------------------------
    # Load all predictions
    # ----------------------------------------------------------
    segment_preds = {}
    for seg in ordered_segments:
        with open(f"{args.pred_dir}/{seg}_p.pkl", "rb") as f:
            segment_preds[seg] = pkl.load(f)

    # ----------------------------------------------------------
    # Load timing annotations
    # ----------------------------------------------------------
    segment_annos = {}
    for seg in ordered_segments:
        with open(f"{args.timing_annos}/{seg}_a.pkl", "rb") as f:
            segment_annos[seg] = pkl.load(f)

    timing_key = f"timing_{args.detector}"

    # ----------------------------------------------------------
    # Reassemble det_annos in dataset order and build
    # frame_id -> (info, anno) lookup for fast filtering
    # ----------------------------------------------------------
    segment_frame_counters = {}
    det_annos = []
    for info in dataset.infos:
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        idx = segment_frame_counters[seg]
        det_annos.append(segment_preds[seg][idx])
        segment_frame_counters[seg] += 1

    for pred, info in zip(det_annos, dataset.infos):
        assert pred['frame_id'] == info['frame_id'], \
            f"Frame ID mismatch: {pred['frame_id']} vs {info['frame_id']}"

    frame_id_to_info = {info['frame_id']: info for info in dataset.infos}
    frame_id_to_anno = {anno['frame_id']: anno for anno in det_annos}

    # ----------------------------------------------------------
    # Pre-load spoof_gt for every frame upfront — one pass over
    # all segment _d.pkl files. This is the only thing we need
    # from the dataset files, and it's the same across all k,
    # so there's no reason to re-load per window.
    # ----------------------------------------------------------
    logger.info("Pre-loading spoof_gt from dataset files...")
    frame_id_to_spoof_gt = {}
    for seg in tqdm(ordered_segments, desc="Loading spoof_gt"):
        seg_dataset_file = f"{args.dataset}/{seg}_d.pkl"
        with open(seg_dataset_file, "rb") as f:
            seg_data = pkl.load(f)
        for frame in seg_data:
            gt = frame.get('spoof_gt', None)
            if gt is not None:
                frame_id_to_spoof_gt[frame['frame_id']] = gt
        del seg_data  # free memory immediately after extracting what we need
    logger.info(f"Loaded spoof_gt for {len(frame_id_to_spoof_gt)} frames")

    # ----------------------------------------------------------
    # Build per-segment block structures
    #   k=0  : frame at block_start - 1  (clean hist, clean obs)
    #   k=1  : frame at block_start + 0  (clean hist, spoofed obs)  [timing==1]
    #   k=2  : frame at block_start + 1  (1 spoofed in hist)        [timing==2]
    #   ...
    #   k=32 : frame at block_start + 31 (31 spoofed in hist)       [timing==2]
    # ----------------------------------------------------------
    NUM_WINDOWS = 33  # 0 through 32 inclusive
    window_frame_ids = {k: [] for k in range(NUM_WINDOWS)}

    for seg in ordered_segments:
        timing_arr = np.array(segment_annos[seg][timing_key])
        seg_preds  = segment_preds[seg]
        block_starts = find_block_starts(timing_arr)

        for bs in block_starts:
            # k=0: frame just before the spoof block
            k0_idx = bs - 1
            if k0_idx >= 0 and timing_arr[k0_idx] == 0:
                frame_id = seg_preds[k0_idx]['frame_id']
                window_frame_ids[0].append((frame_id, seg, k0_idx))

            # k=1..32: frames within the spoof block
            for k in range(1, NUM_WINDOWS):
                offset_idx = bs + (k - 1)
                if offset_idx >= len(timing_arr):
                    break
                expected_timing = 1 if k == 1 else 2
                if timing_arr[offset_idx] != expected_timing:
                    break  # block ended early (segment boundary)
                frame_id = seg_preds[offset_idx]['frame_id']
                window_frame_ids[k].append((frame_id, seg, offset_idx))

    for k in range(NUM_WINDOWS):
        logger.info(f"Window k={k:2d}: {len(window_frame_ids[k])} frames")

    # ----------------------------------------------------------
    # Evaluate each window
    # ----------------------------------------------------------
    results_list = []
    original_infos = dataset.infos

    for k in tqdm(range(NUM_WINDOWS), desc="Evaluating windows"):
        entries = window_frame_ids[k]
        if len(entries) == 0:
            logger.warning(f"Window k={k}: no frames found, skipping")
            results_list.append(None)
            continue

        fids          = [e[0] for e in entries]
        segs          = [e[1] for e in entries]
        local_indices = [e[2] for e in entries]

        filtered_infos = [frame_id_to_info[fid] for fid in fids if fid in frame_id_to_info]
        filtered_annos = [frame_id_to_anno[fid] for fid in fids if fid in frame_id_to_anno]

        if len(filtered_infos) == 0:
            logger.warning(f"Window k={k}: no matching infos, skipping")
            results_list.append(None)
            continue

        for pred, info in zip(filtered_annos, filtered_infos):
            assert pred['frame_id'] == info['frame_id']

        # ----------------------------------------------------------
        # ASR: for spoofed frames (k>=1), check for vehicle detection
        # at the spoof ground truth location (IoU >= 0.7).
        # spoof_gt is looked up directly from the preloaded dict —
        # no disk IO here anymore.
        # ----------------------------------------------------------
        asr_records = []
        total_spoofed = 0
        if k >= 1:
            for fid, seg, loc_idx in zip(fids, segs, local_indices):
                spoof_gt = frame_id_to_spoof_gt.get(fid, None)
                if spoof_gt is None:
                    continue
                total_spoofed += 1
                frame_pred = frame_id_to_anno[fid]
                pred_boxes = frame_pred.get('boxes_lidar', None)
                if pred_boxes is None or len(pred_boxes) == 0:
                    asr_records.append({
                        'segment': seg, 'local_idx': loc_idx,
                        'frame_id': fid, 'iou': 0.0, 'score': 0.0,
                    })
                    continue

                pred_labels = frame_pred.get('pred_labels', np.array([]))
                scores      = frame_pred.get('score',       np.array([]))

                target_class = spoof_gt[-1]  # last element is class label
                iou_thresh = 0.7 if target_class == 1 else 0.5  # 0.7 for vehicles, 0.5 for pedestrians/cyclists
                
                target_mask = (np.array(pred_labels) == target_class)
                if not np.any(target_mask):
                    asr_records.append({
                        'segment': seg, 'local_idx': loc_idx,
                        'frame_id': fid, 'iou': 0.0, 'score': 0.0,
                    })
                    continue

                target_boxes  = pred_boxes[target_mask]
                target_scores = np.array(scores)[target_mask]

                gt_tensor   = torch.tensor(spoof_gt[:7], dtype=torch.float32).unsqueeze(0).cuda()
                pred_tensor = torch.tensor(target_boxes[:, :7], dtype=torch.float32).cuda()

                iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_tensor, gt_tensor)  # (N,1)
                max_iou, best_obj = iou[:, 0].max(0)
                max_iou = max_iou.item()

                if not (max_iou >= iou_thresh):
                    asr_records.append({
                        'segment':   seg,
                        'local_idx': loc_idx,
                        'frame_id':  fid,
                        'iou':       max_iou,
                        'score':     float(target_scores[best_obj.item()]),
                    })

        dataset.infos = filtered_infos

        _, result_dict = dataset.evaluation(
            filtered_annos,
            cfg.CLASS_NAMES,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

        asr = len(asr_records) / total_spoofed if k >= 1 and total_spoofed > 0 else None

        entry = {
            'k':           k,
            'num_frames':  len(filtered_infos),
            'ap':          result_dict,
            'asr':         asr,
            'asr_records': asr_records,
        }
        results_list.append(entry)

        asr_str = f"ASR={asr:.4f}" if asr is not None else "ASR=N/A (clean baseline)"
        logger.info(f"k={k:2d} | frames={len(filtered_infos):4d} | {asr_str} | "
                    f"Veh_L1={result_dict.get('OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP', float('nan')):.4f} | "
                    f"Ped_L1={result_dict.get('OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP', float('nan')):.4f} | "
                    f"Cyc_L1={result_dict.get('OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP', float('nan')):.4f}")

    dataset.infos = original_infos

    with open(args.metrics_out, "wb") as f:
        pkl.dump(results_list, f)

    logger.info(f"History curve saved to {args.metrics_out}")


def main():
    args = parse_args()
    logger = setup_logger(args.log_file)
    try:
        run_evaluations(args, logger)
    except Exception as e:
        logger.error("===== EVALUATION FAILED =====")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()