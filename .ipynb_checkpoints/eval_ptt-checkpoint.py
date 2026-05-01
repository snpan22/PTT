import argparse
import pickle
import logging
import json
from pathlib import Path
import traceback
import sys
from tqdm import tqdm
from pcdet.utils import common_utils

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader


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
    parser.add_argument('--cfg_file', required=True)
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
    pcdet_logger = common_utils.create_logger()

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        training=False,
        logger=pcdet_logger
    )

    pred_dir = args.pred_dir

    segment_preds = {}
    unique_segment_names = dataset.seq_name_to_infos.keys()
    for segment_name in unique_segment_names:
        with open(f"{pred_dir}/{segment_name}_p.pkl", "rb") as f:
            segment_preds[segment_name] = pickle.load(f)
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

    logger.info("Starting Waymo evaluation")

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