import os
import pickle as pkl
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def to_numpy(x):
    """Convert tensors/lists to numpy arrays without changing numpy inputs."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def in_azimuth_wedge(gt_boxes, az_center, az_width_rad):
    """
    Return a boolean mask for GT boxes whose centers lie inside the azimuth wedge.

    Parameters
    ----------
    gt_boxes : (N, D)
    az_center : float
        Sector center in radians, wrapped to [-pi, pi].
    az_width_rad : float
        Full sector width in radians. Internally we compare to half-width.
    """
    if gt_boxes is None or len(gt_boxes) == 0:
        return np.zeros(0, dtype=bool)

    centers = gt_boxes[:, :3]
    x, y = centers[:, 0], centers[:, 1]
    theta = wrap_to_pi(np.arctan2(y, x))
    delta = wrap_to_pi(theta - az_center)
    half_width = az_width_rad / 2.0
    return np.abs(delta) <= half_width


def get_pred_boxes_labels_scores(frame_pred):
    """
    Read predictions from either:
      - OpenPCDet/MSF/PTT result dicts: boxes_lidar / pred_labels / score
      - custom per-frame dicts: pred_boxes / pred_labels / pred_scores

    Always returns numpy arrays (or None for boxes if absent).
    """
    pred_boxes = frame_pred.get('boxes_lidar', frame_pred.get('pred_boxes', None))
    pred_labels = frame_pred.get('pred_labels', np.array([]))
    pred_scores = frame_pred.get('score', frame_pred.get('pred_scores', np.array([])))

    pred_boxes = to_numpy(pred_boxes)
    pred_labels = to_numpy(pred_labels)
    pred_scores = to_numpy(pred_scores)

    if pred_labels is None:
        pred_labels = np.array([])
    if pred_scores is None:
        pred_scores = np.array([])

    return pred_boxes, np.array(pred_labels), np.array(pred_scores)


def matched_score_if_detected(pred_boxes, pred_labels, pred_scores, gt_box, target_class, threshold):
    """
    For a single GT box, determine whether the detector produced a *valid* detection
    of the same class with IoU >= threshold.

    Returns
    -------
    detected : bool
    score : float | None
        Score of the best valid matched prediction. None if no valid match exists.
    max_iou : float
        Best IoU among same-class predictions. 0 if no same-class predictions exist.

    Important semantic choice:
      - Sub-threshold overlaps are treated as *not detected*.
      - Their scores are not used in downstream score analysis.
    """
    if pred_boxes is None or len(pred_boxes) == 0:
        return False, None, 0.0

    class_mask = (pred_labels == target_class)
    if not np.any(class_mask):
        return False, None, 0.0

    boxes_same_class = pred_boxes[class_mask]
    scores_same_class = pred_scores[class_mask]

    gt_tensor = torch.tensor(gt_box[:7], dtype=torch.float32).unsqueeze(0).cuda()
    pred_tensor = torch.tensor(boxes_same_class[:, :7], dtype=torch.float32).cuda()

    iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_tensor, gt_tensor)  # (N, 1)
    best_idx = torch.argmax(iou[:, 0]).item()
    max_iou = float(iou[best_idx, 0].item())

    if max_iou < threshold:
        return False, None, max_iou

    return True, float(scores_same_class[best_idx]), max_iou


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg.clear()

    CFG_FILE = 'tools/cfgs/waymo_models/ptt_32frames.yaml'
    cfg_from_yaml_file(CFG_FILE, cfg)
    cfg.TAG = Path(CFG_FILE).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'

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
    logger.info(f'Test set length: {len(dataset)}')

    # Touch the loader once so CUDA / env issues surface early.
    data_iter = iter(test_loader)
    batch_dict = next(data_iter)
    from pcdet.models import load_data_to_gpu  # imported lazily to avoid unused import noise
    load_data_to_gpu(batch_dict)

    ordered_segments = [info['point_cloud']['lidar_sequence'] for info in dataset.infos]
    ordered_segments = list(dict.fromkeys(ordered_segments))

    pred_dirs = ['preds_final', 'preds_final_msf4', 'preds_final_msf8', 'preds_final_ptt']
    dataset_dir = 'datasets'

    dataset_names = [name for name in os.listdir(dataset_dir) if 'removal' in name]

    pred_names = []
    for dir_ in pred_dirs:
        detector_preds = os.listdir(dir_)
        if dir_ == 'preds_final':
            detector_preds = [name for name in detector_preds if 'removal' in name and name.startswith('cp_')]
        else:
            detector_preds = [name for name in detector_preds if 'removal' in name]
        print(detector_preds)
        pred_names.extend(detector_preds)

    # -------------------------------------------------------------------------
    # Clean predictions for paired score analysis
    # -------------------------------------------------------------------------
    clean_preds_filepaths = [
        "/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_singleframe_waymo/default/eval/eval_with_train/epoch_30/val/result.pkl",
        "/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/val/default/result_fixed.pkl",
        "/storage/project/r-gchou3-0/spanse30/MSF/output/cfgs/waymo_models/msf_4frames/default/eval/eval_with_train/epoch_6/val/result.pkl",
        "/storage/project/r-gchou3-0/spanse30/MSF/output/cfgs/waymo_models/msf_8frames/default/eval/eval_with_train/epoch_6/val/result.pkl",
        "/storage/home/hcoda1/9/spanse30/scratch/PTT/output/storage/scratch1/9/spanse30/PTT/tools/cfgs/waymo_models/ptt_32frames/default/eval/eval_with_train/epoch_6/val/result.pkl",
    ]
    clean_preds = {'cp': None, 'cp4f': None, 'msf4': None, 'msf8': None, 'ptt': None}
    for path, detector in zip(clean_preds_filepaths, clean_preds.keys()):
        with open(path, "rb") as f:
            preds = pkl.load(f)
        # Normalize keys to plain str so lookup works across np.str_ / Python str.
        clean_pred_map = {str(p['frame_id']): p for p in preds}
        clean_preds[detector] = clean_pred_map

    asr_annos = []

    for dataset_name in dataset_names:
        search_term = dataset_name
        subset = [s for s in pred_names if search_term in s]
        subset.append(f"cp4f_{search_term}")

        order = {'cp4f': 0, 'cp': 1, 'msf4': 2, 'msf8': 3, 'ptt': 4}
        subset = sorted(subset, key=lambda x: order[x.split('_')[0]])

        print(subset)
        dataset_annos_dir = 'annos/' + dataset_name
        print(dataset_annos_dir)

        for detector in subset:
            detector_name = detector.split('_', 1)[0]
            clean_preds_detector = clean_preds[detector_name]

            pred_dir = 'preds_final' if detector.startswith('cp_') else 'preds_final_' + detector.split('_', 1)[0]

            asr_annos_det_mode = {'name': detector}
            segment_preds = {}
            segment_annos = {}

            if not detector.startswith('cp4f'):
                for segment_name in ordered_segments:
                    with open(f"{pred_dir}/{detector}/{segment_name}_p.pkl", "rb") as f:
                        segment_preds[segment_name] = pkl.load(f)

            for segment_name in ordered_segments:
                with open(f"{dataset_annos_dir}/{segment_name}_a.pkl", "rb") as f:
                    segment_annos[segment_name] = pkl.load(f)

            print(f"{pred_dir}/{detector}/")

            current_segment = None
            seg_dataset = None

            # -----------------------------------------------------------------
            # Metric counters
            # -----------------------------------------------------------------
            total_target_frames = 0
            num_0_preds = 0
            num_targets_hid = 0
            num_collateral_hid = 0
            total_collateral_boxes = 0
            num_frames_collateral_hid = 0

            per_frame_removal_fraction = []
            collateral_per_frame = []

            # -----------------------------------------------------------------
            # Score lists (NEW SEMANTICS)
            #
            # These are now paired score lists for *surviving detections only*.
            # We only append when:
            #   1) spoofed detector still detects the GT object (attack unsuccessful), and
            #   2) clean detector also detects that same GT object.
            #
            # This avoids storing meaningless scores for sub-threshold matches.
            # -----------------------------------------------------------------
            scores_target_object = []
            scores_target_clean = []
            scores_collateral_objects = []
            scores_collateral_clean = []

            skipped_target_pairs_no_clean = 0
            skipped_collateral_pairs_no_clean = 0

            for info in dataset.infos:
                frame_id = info['frame_id']
                parts = frame_id.rsplit('_', 1)
                segment = parts[0]
                i_seg = int(parts[1])

                if current_segment != segment:
                    seg_dataset_file = f"{dataset_dir}/{dataset_name}/{segment}_d.pkl"
                    with open(seg_dataset_file, "rb") as f:
                        seg_dataset = pkl.load(f)
                    current_segment = segment

                assert frame_id == seg_dataset[i_seg]['frame_id']

                frame = seg_dataset[i_seg]
                frame_anno = segment_annos[current_segment]['spoof_annos'][i_seg]

                if not detector.startswith('cp4f'):
                    frame_pred = segment_preds[segment][i_seg]
                else:
                    # cp4f predictions are already stored inside the custom dataset entry.
                    frame_pred = {
                        'boxes_lidar': frame['pred_boxes'],
                        'score': frame['pred_scores'],
                        'pred_labels': frame['pred_labels'],
                    }

                frame_pred_clean = clean_preds_detector[str(frame_id)]
                gt_spoof = frame['spoof_gt']

                # Evaluate only frames where the current frame was actually attacked.
                if gt_spoof is None or not frame_anno['active_this_frame']:
                    continue

                total_target_frames += 1

                pred_boxes, pred_labels, pred_scores = get_pred_boxes_labels_scores(frame_pred)
                pred_boxes_clean, pred_labels_clean, pred_scores_clean = get_pred_boxes_labels_scores(frame_pred_clean)
                gt_boxes = frame['gt_boxes']

                if pred_boxes is None or len(pred_boxes) == 0:
                    num_0_preds += 1

                # -------------------------------------------------------------
                # Target metrics
                # -------------------------------------------------------------
                target_class = int(gt_spoof[-1])
                target_threshold = 0.7 if target_class == 1 else 0.5

                spoof_detected, spoof_score, _ = matched_score_if_detected(
                    pred_boxes, pred_labels, pred_scores,
                    gt_spoof, target_class, target_threshold
                )

                if not spoof_detected:
                    num_targets_hid += 1
                else:
                    # Only keep paired target scores when BOTH spoofed and clean
                    # detections are valid matches to the same GT target.
                    clean_detected, clean_score, _ = matched_score_if_detected(
                        pred_boxes_clean, pred_labels_clean, pred_scores_clean,
                        gt_spoof, target_class, target_threshold
                    )
                    if clean_detected:
                        scores_target_object.append(spoof_score)
                        scores_target_clean.append(clean_score)
                    else:
                        skipped_target_pairs_no_clean += 1

                # -------------------------------------------------------------
                # Collateral set: GT objects inside the attacked sector, excluding
                # the primary spoof target so ASR and collateral metrics remain
                # separate.
                # -------------------------------------------------------------
                az_center = frame_anno['az_center_rad']
                az_width = np.deg2rad(frame_anno['az_width_deg'])

                gt_boxes_in_sector = gt_boxes[in_azimuth_wedge(gt_boxes, az_center, az_width)]
                mask_not_spoof = ~np.all(np.isclose(gt_boxes_in_sector, gt_spoof, atol=1e-6), axis=1)
                gt_boxes_collateral = gt_boxes_in_sector[mask_not_spoof]

                collateral_per_frame.append(len(gt_boxes_collateral))
                num_collateral_hid_in_frame = 0

                for collateral_box in gt_boxes_collateral:
                    total_collateral_boxes += 1

                    collateral_class = int(collateral_box[-1])
                    collateral_threshold = 0.7 if collateral_class == 1 else 0.5

                    spoof_detected, spoof_score, _ = matched_score_if_detected(
                        pred_boxes, pred_labels, pred_scores,
                        collateral_box, collateral_class, collateral_threshold
                    )

                    if not spoof_detected:
                        num_collateral_hid += 1
                        num_collateral_hid_in_frame += 1
                    else:
                        # Same paired-score rule as targets: only compare scores when
                        # both spoofed and clean predictions are valid detections.
                        clean_detected, clean_score, _ = matched_score_if_detected(
                            pred_boxes_clean, pred_labels_clean, pred_scores_clean,
                            collateral_box, collateral_class, collateral_threshold
                        )
                        if clean_detected:
                            scores_collateral_objects.append(spoof_score)
                            scores_collateral_clean.append(clean_score)
                        else:
                            skipped_collateral_pairs_no_clean += 1

                if num_collateral_hid_in_frame > 0:
                    num_frames_collateral_hid += 1

                removal_fraction = 0.0 if len(gt_boxes_collateral) == 0 else num_collateral_hid_in_frame / len(gt_boxes_collateral)
                per_frame_removal_fraction.append(removal_fraction)

            # -----------------------------------------------------------------
            # Aggregate metrics
            # -----------------------------------------------------------------
            collateral_removal_rate = 0.0 if total_collateral_boxes == 0 else num_collateral_hid / total_collateral_boxes
            sector_failure_rate = 0.0 if total_target_frames == 0 else num_frames_collateral_hid / total_target_frames
            per_frame_removal_fraction_avg = float(np.mean(per_frame_removal_fraction)) if len(per_frame_removal_fraction) > 0 else 0.0
            target_asr = 0.0 if total_target_frames == 0 else num_targets_hid / total_target_frames
            zero_pred_rate = 0.0 if total_target_frames == 0 else num_0_preds / total_target_frames

            # These lengths should be equal by construction because we only append
            # paired spoof/clean scores together.
            assert len(scores_target_object) == len(scores_target_clean)
            assert len(scores_collateral_objects) == len(scores_collateral_clean)

            asr_annos_det_mode['crr'] = collateral_removal_rate
            asr_annos_det_mode['sfr'] = sector_failure_rate
            asr_annos_det_mode['mean_pfr'] = per_frame_removal_fraction_avg
            asr_annos_det_mode['pfr'] = per_frame_removal_fraction
            asr_annos_det_mode['target_asr'] = target_asr
            asr_annos_det_mode['n_targets'] = total_target_frames
            asr_annos_det_mode['n_collateral'] = total_collateral_boxes
            asr_annos_det_mode['n_frames_collateral_hid'] = num_frames_collateral_hid
            asr_annos_det_mode['collateral_per_frame'] = collateral_per_frame
            asr_annos_det_mode['zero_pred_rate'] = zero_pred_rate

            # Score outputs now mean: paired scores for surviving detections only.
            asr_annos_det_mode['scores_target'] = scores_target_object
            asr_annos_det_mode['scores_target_clean'] = scores_target_clean
            asr_annos_det_mode['scores_collateral'] = scores_collateral_objects
            asr_annos_det_mode['scores_collateral_clean'] = scores_collateral_clean
            asr_annos_det_mode['n_target_score_pairs'] = len(scores_target_object)
            asr_annos_det_mode['n_collateral_score_pairs'] = len(scores_collateral_objects)
            asr_annos_det_mode['skipped_target_pairs_no_clean'] = skipped_target_pairs_no_clean
            asr_annos_det_mode['skipped_collateral_pairs_no_clean'] = skipped_collateral_pairs_no_clean

            print(f"CRR: {collateral_removal_rate}")
            print(f"sector failure rate: {sector_failure_rate}")
            print(f"average per frame removal fraction: {per_frame_removal_fraction_avg}")
            print(f"target asr: {target_asr}")
            print(f"target score pairs: {len(scores_target_object)}")
            print(f"collateral score pairs: {len(scores_collateral_objects)}")

            asr_annos.append(asr_annos_det_mode)

    with open("asr/asr_pr_newscorelogic.pkl", "wb") as f:
        pkl.dump(asr_annos, f)
