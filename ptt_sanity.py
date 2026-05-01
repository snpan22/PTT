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
from tqdm import tqdm

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
    logger = logging.getLogger("eval_clean_sanity")
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
    parser.add_argument('--clean_result_pkl', required=True,
                        help='Clean CenterPoint result.pkl — used to build proposal history')
    parser.add_argument('--cfg_file',  required=True)
    parser.add_argument('--ckpt',      required=True)
    parser.add_argument('--pred_dir',  required=True)
    parser.add_argument('--workers',   type=int, default=4)
    parser.add_argument('--log_file',  default='eval_clean_sanity.log')
    parser.add_argument('--metrics_out', default='metrics_clean_sanity.json')
    return parser.parse_args()


# ------------------------------------------------------------
# Load clean result.pkl into per-segment ordered structures
# ------------------------------------------------------------
def load_clean_proposals(clean_result_pkl, dataset, logger):
    """
    Loads clean CenterPoint result.pkl and organises proposals into
    per-segment lists ordered by local frame index.

    Returns:
        seg_proposals: dict[segment] -> list of dicts, one per frame:
            {
              'frame_id':    str,
              'pred_boxes':  tensor (N, 9),
              'pred_scores': tensor (N,),
              'pred_labels': tensor (N,),
            }
    """
    logger.info(f"Loading clean proposals from {clean_result_pkl}")
    with open(clean_result_pkl, 'rb') as f:
        result = pkl.load(f)

    frame_to_preds = {d['frame_id']: d for d in result}

    def to_tensor(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(dtype).cpu()
        return torch.tensor(x, dtype=dtype)

    seg_proposals = {}
    seg_counters  = {}

    for info in dataset.infos:
        seg      = info['point_cloud']['lidar_sequence']
        frame_id = info['frame_id']

        seg_counters.setdefault(seg, 0)
        seg_counters[seg] += 1

        if seg not in seg_proposals:
            seg_proposals[seg] = []

        if frame_id in frame_to_preds:
            d      = frame_to_preds[frame_id]
            boxes  = to_tensor(d['pred_boxes'])                          # (N, 9)
            scores = to_tensor(d.get('pred_scores',
                                     torch.zeros(len(boxes))))           # (N,)
            labels = to_tensor(d.get('pred_labels',
                                     torch.zeros(len(boxes))),
                               dtype=torch.long)                         # (N,)
        else:
            # Frame missing from clean result — insert empty proposals
            logger.warning(f"Frame {frame_id} not found in clean result pkl — using empty proposals")
            boxes  = torch.zeros((0, 9), dtype=torch.float32)
            scores = torch.zeros(0,      dtype=torch.float32)
            labels = torch.zeros(0,      dtype=torch.long)

        seg_proposals[seg].append({
            'frame_id':    frame_id,
            'pred_boxes':  boxes,
            'pred_scores': scores,
            'pred_labels': labels,
        })

    logger.info(f"Organised clean proposals for {len(seg_proposals)} segments")
    return seg_proposals


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_evaluations(args, logger):
    logger.info("Loading config...")
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'
    logger.info(f'Loaded cfg from {args.cfg_file}')

    dataset, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )
    logger.info(f'Test set length: {len(dataset)}')

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )
    logger.info(f'Loading checkpoint from: {args.ckpt}')
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # Build global_idx -> (segment, local_idx)
    global_to_segment = {}
    seg_counters = {}
    for i, info in enumerate(dataset.infos):
        seg = info['point_cloud']['lidar_sequence']
        seg_counters.setdefault(seg, 0)
        global_to_segment[i] = (seg, seg_counters[seg])
        seg_counters[seg] += 1

    # Load clean proposals organised by segment
    seg_clean_proposals = load_clean_proposals(
        args.clean_result_pkl, dataset, logger
    )

    save_dir_preds = args.pred_dir
    os.makedirs(save_dir_preds, exist_ok=True)
    completed = set(f[:-4] for f in os.listdir(save_dir_preds))

    history = 32
    segment_preds_list = []
    current_segment    = None

    try:
        for (i_ptt, batch) in enumerate(test_loader):

            segment = batch['frame_id'][0].rsplit("_", 1)[0]
            seg_idx = global_to_segment[i_ptt][1]

            if f"{segment}_p" in completed:
                continue

            if current_segment != segment:
                if current_segment is not None:
                    pred_path = f"{save_dir_preds}/{current_segment}_p.pkl"
                    with open(pred_path, "wb") as f:
                        pkl.dump(segment_preds_list, f)
                    logger.info("Saved preds for: %s", current_segment)
                segment_preds_list = []
                current_segment    = segment
                logger.info("Processing segment: %s", current_segment)

            # Build 32-frame proposal history from clean result.pkl
            clean_proposals  = seg_clean_proposals.get(segment, [])
            past_frames_list = [max(seg_idx - i, 0) for i in range(history)]

            preds_len = [
                clean_proposals[frame]['pred_boxes'].shape[0]
                if frame < len(clean_proposals) else 0
                for frame in past_frames_list
            ]
            max_preds = max(preds_len) if max(preds_len) > 0 else 1

            past_boxes  = []
            past_scores = []
            past_labels = []

            for frame_seg in past_frames_list:
                if frame_seg < len(clean_proposals):
                    preds  = clean_proposals[frame_seg]['pred_boxes']   # (N, 9)
                    scores = clean_proposals[frame_seg]['pred_scores']  # (N,)
                    labels = clean_proposals[frame_seg]['pred_labels']  # (N,)
                else:
                    preds  = torch.zeros((0, 9), dtype=torch.float32)
                    scores = torch.zeros(0,      dtype=torch.float32)
                    labels = torch.zeros(0,      dtype=torch.long)

                pad_size = max_preds - preds.shape[0]
                past_boxes.append( F.pad(preds,  (0, 0, 0, pad_size), value=0))
                past_scores.append(F.pad(scores, (0, pad_size),        value=0))
                past_labels.append(F.pad(labels, (0, pad_size),        value=0))

            past_boxes  = torch.stack(past_boxes,  dim=0).detach().cpu().numpy()
            past_scores = torch.stack(past_scores, dim=0).detach().cpu().numpy()
            past_labels = torch.stack(past_labels, dim=0).detach().cpu().numpy()

            # points come from the clean pcdet dataset — no override
            dict_mod = dataset[i_ptt].copy()
            dict_mod['roi_boxes']  = past_boxes
            dict_mod['roi_scores'] = past_scores
            dict_mod['roi_labels'] = past_labels

            dict_mod  = helpers_ptt.inject_gt_names(dict_mod, dataset.class_names)
            dict_mod  = dataset.prepare_data(dict_mod)
            batch_mod = dataset.collate_batch([dict_mod])
            load_data_to_gpu(batch_mod)

            with torch.no_grad():
                pred_dicts, _ = model(batch_mod)

            annos = dataset.generate_prediction_dicts(
                batch_mod, pred_dicts, cfg.CLASS_NAMES
            )
            segment_preds_list += annos

    except Exception:
        logger.error("===== EXCEPTION =====")
        logger.error(traceback.format_exc())
        raise

    finally:
        if current_segment is not None and len(segment_preds_list) > 0:
            pred_path = f"{save_dir_preds}/{current_segment}_p.pkl"
            with open(pred_path, "wb") as f:
                pkl.dump(segment_preds_list, f)
            logger.info("Saved preds for: %s", current_segment)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    segment_preds = {}
    for segment_name in dataset.seq_name_to_infos.keys():
        p = f"{save_dir_preds}/{segment_name}_p.pkl"
        if os.path.exists(p):
            with open(p, "rb") as f:
                segment_preds[segment_name] = pkl.load(f)

    det_annos = []
    seg_counters = {}
    for info in dataset.infos:
        segment = info['point_cloud']['lidar_sequence']
        seg_counters.setdefault(segment, 0)
        idx = seg_counters[segment]
        seg_counters[segment] += 1
        if segment in segment_preds:
            det_annos.append(segment_preds[segment][idx])

    for pred, info in zip(det_annos, dataset.infos):
        assert pred['frame_id'] == info['frame_id']

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
    logger.info(json.dumps(result_dict, indent=2, default=float))

    with open(args.metrics_out, "w") as f:
        json.dump(result_dict, f, indent=2, default=float)
    logger.info(f"Metrics saved to {args.metrics_out}")


def main():
    args   = parse_args()
    logger = setup_logger(args.log_file)

    try:
        run_evaluations(args, logger)
    except Exception as e:
        logger.error("===== EVALUATION FAILED =====")
        logger.error(str(e))
        logger.error("\nFull traceback:\n")
        logger.error(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()