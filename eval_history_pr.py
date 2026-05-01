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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--pred_dir', required=True,
                        help='Directory containing <segment>_p.pkl prediction files')
    parser.add_argument('--timing_annos', required=True,
                        help='Directory containing <segment>_a.pkl annotation files')
    parser.add_argument('--dataset', required=True,
                        help='Directory containing <segment>_d.pkl spoofed dataset files')
    parser.add_argument('--history_len', type=int, default=32,
                        help='History window length. Use 32 for PTT, 4/8 for MSF variants.')
    parser.add_argument('--metrics_out', default='history_curve.pkl')
    parser.add_argument('--log_file', default='eval_history_curve.log')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def frame_fired(meta):
    return meta is not None and bool(meta.get('active_this_frame', False))
def get_pred_boxes_labels_scores(frame_pred):
    pred_boxes = frame_pred.get('boxes_lidar', frame_pred.get('pred_boxes', None))
    pred_labels = frame_pred.get('pred_labels', np.array([]))
    pred_scores = frame_pred.get('score', frame_pred.get('pred_scores', np.array([])))
    return pred_boxes, np.array(pred_labels), np.array(pred_scores)

def safe_pkl_load(path):
    try:
        with open(path, "rb") as f:
            return pkl.load(f)
    except RuntimeError as e:
        if "CUDA" in str(e):
            with open(path, "rb") as f:
                return torch.load(f, map_location='cpu')
        raise

def build_window_frame_ids_from_spoof_annos(segment_annos, segment_preds, ordered_segments, history_len):
    """
    k = number of frames in the last `history_len` frames (including current)
        for which active_this_frame == True.

    k = 0  -> clean current frame + no fired frames in lookback window
    k >= 1 -> current frame fired, and total fired count in the window is k
    """
    num_windows = history_len + 1
    window_frame_ids = {k: [] for k in range(num_windows)}

    for seg in ordered_segments:
        seg_preds_local = segment_preds[seg]
        spoof_annos = segment_annos[seg]['spoof_annos']

        n = len(seg_preds_local)
        fired_flags = np.zeros(n, dtype=np.int32)

        for local_idx in range(n):
            meta = spoof_annos.get(local_idx, None)
            fired_flags[local_idx] = 1 if frame_fired(meta) else 0

        prefix = np.concatenate([[0], np.cumsum(fired_flags)])

        for local_idx in range(n):
            left = max(0, local_idx - history_len + 1)
            fired_count = int(prefix[local_idx + 1] - prefix[left])
            cur_fired = bool(fired_flags[local_idx])

            frame_id = seg_preds_local[local_idx]['frame_id']

            if cur_fired:
                # current frame fired; bin by actual fired count in window
                window_frame_ids[fired_count].append((frame_id, seg, local_idx))
            else:
                # clean baseline only if entire lookback window is clean
                if fired_count == 0:
                    window_frame_ids[0].append((frame_id, seg, local_idx))

    return window_frame_ids


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
    # Load predictions and annotations
    # ----------------------------------------------------------
    segment_preds = {}
    segment_annos = {}

    for seg in ordered_segments:
        with open(f"{args.pred_dir}/{seg}_p.pkl", "rb") as f:
            segment_preds[seg] = pkl.load(f)
        with open(f"{args.timing_annos}/{seg}_a.pkl", "rb") as f:
            segment_annos[seg] = pkl.load(f)

    # ----------------------------------------------------------
    # Reassemble det_annos in dataset order
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
    # Pre-load spoof_gt
    # ----------------------------------------------------------
    logger.info("Pre-loading spoof_gt from dataset files...")
    frame_id_to_spoof_gt = {}
    for seg in tqdm(ordered_segments, desc="Loading spoof_gt"):
        seg_dataset_file = f"{args.dataset}/{seg}_d.pkl"
        seg_data = safe_pkl_load(seg_dataset_file)
        for frame in seg_data:
            gt = frame.get('spoof_gt', None)
            if gt is not None:
                frame_id_to_spoof_gt[frame['frame_id']] = gt
        del seg_data
    logger.info(f"Loaded spoof_gt for {len(frame_id_to_spoof_gt)} frames")

    # ----------------------------------------------------------
    # NEW: build windows from actual fired frames
    # ----------------------------------------------------------
    num_windows = args.history_len + 1
    window_frame_ids = build_window_frame_ids_from_spoof_annos(
        segment_annos=segment_annos,
        segment_preds=segment_preds,
        ordered_segments=ordered_segments,
        history_len=args.history_len
    )

    for k in range(num_windows):
        logger.info(f"Window k={k:2d}: {len(window_frame_ids[k])} frames")

    # ----------------------------------------------------------
    # Evaluate each window
    # ----------------------------------------------------------
    results_list = []
    original_infos = dataset.infos

    for k in tqdm(range(num_windows), desc="Evaluating windows"):
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
        # ASR conditioned on current frame attack actually firing
        # ----------------------------------------------------------
        asr_records = []
        total_fired = 0

        if k >= 1:
            for fid, seg, loc_idx in zip(fids, segs, local_indices):
                spoof_gt = frame_id_to_spoof_gt.get(fid, None)
                meta = segment_annos[seg]['spoof_annos'].get(loc_idx, None)

                # condition on actual fired frame
                if spoof_gt is None or meta is None or not meta.get('active_this_frame', False):
                    continue

                total_fired += 1
                frame_pred = frame_id_to_anno[fid]
                # pred_boxes = frame_pred.get('boxes_lidar', None)
                pred_boxes, pred_labels, scores = get_pred_boxes_labels_scores(frame_pred)
                #no predictions at all -> failure to detect
                if pred_boxes is None or len(pred_boxes) == 0:
                    continue

                # pred_labels = frame_pred.get('pred_labels', np.array([]))
                # scores      = frame_pred.get('score', np.array([]))

                target_class = int(spoof_gt[-1])
                iou_thresh = 0.7 if target_class == 1 else 0.5

                # no objects detected in target class -> failure to detect
                target_mask = (np.array(pred_labels) == target_class)
                if not np.any(target_mask):
                    continue

                target_boxes  = pred_boxes[target_mask]
                target_scores = np.array(scores)[target_mask]

                gt_tensor   = torch.tensor(spoof_gt[:7], dtype=torch.float32).unsqueeze(0).cuda()
                pred_tensor = torch.tensor(target_boxes[:, :7], dtype=torch.float32).cuda()

                iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_tensor, gt_tensor)
                max_iou, best_obj = iou[:, 0].max(0)
                max_iou = max_iou.item()

                if max_iou >= iou_thresh:
                    asr_records.append({
                        'segment': seg,
                        'local_idx': loc_idx,
                        'frame_id': fid,
                        'iou': max_iou,
                        'score': float(target_scores[best_obj.item()]),
                        'active_this_frame': True,
                        'n_points_removed': int(meta.get('n_points_removed', 0)),
                    })

        dataset.infos = filtered_infos

        _, result_dict = dataset.evaluation(
            filtered_annos,
            cfg.CLASS_NAMES,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

        asr = 1 - (len(asr_records) / total_fired) if k >= 1 and total_fired > 0 else None

        entry = {
            'k': k,
            'num_frames': len(filtered_infos),
            'ap': result_dict,
            'asr': asr,
            'num_fired_current_frames': total_fired,
            'asr_records': asr_records,
        }
        results_list.append(entry)

        asr_str = f"ASR={asr:.4f}" if asr is not None else "ASR=N/A (clean baseline)"
        logger.info(
            f"k={k:2d} | frames={len(filtered_infos):4d} | fired={total_fired:4d} | {asr_str} | "
            f"Veh_L1={result_dict.get('OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP', float('nan')):.4f} | "
            f"Ped_L1={result_dict.get('OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP', float('nan')):.4f} | "
            f"Cyc_L1={result_dict.get('OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP', float('nan')):.4f}"
        )

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 