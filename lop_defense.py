"""
LOP defense at inference time: wrap a detector's predictions and eliminate
those whose bounding boxes have too many low-objectness pillars.

Updated from v1 with:
- pillar_thresh: configurable per-pillar binarization threshold (use the
  best_thresh reported by training, e.g. ~0.47, instead of hardcoded 0.5).
- class_filter: only apply defense to predictions of specified class labels
  (the paper's LOP is trained on Vehicle pillars only -- applying it to
  pedestrian/cyclist boxes would wrongly eliminate them).
- keep_if_no_pillars: configurable behavior when a box has no qualifying
  pillars (default False to match paper's "no supporting evidence" stance).
- RNG moved to __init__ (avoid recreating per pillar).
"""

from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import Polygon, box as shapely_box
from pointnet_lop import PointNetLOP

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _rotated_box_to_polygon(cx, cy, dx, dy, heading):
    """
    Same convention as generate_lop_training_data.py.
    Returns a shapely Polygon for a rotated 2D box footprint.
    """
    hx = 0.5 * dx
    hy = 0.5 * dy
    corners = np.array(
        [[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]],
        dtype=np.float32,
    )
    c = np.cos(heading)
    s = np.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = corners @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return Polygon(rotated)


def _pillar_rotated_box_iou(pillar_bounds, rotated_poly):
    """
    Exact 2D IoU between pillar AABB and rotated box polygon.
    Matches training data generation semantics.
    """
    px_min, py_min, px_max, py_max = pillar_bounds
    pillar_poly = shapely_box(px_min, py_min, px_max, py_max)
    inter = pillar_poly.intersection(rotated_poly).area
    if inter <= 0.0:
        return 0.0
    union = pillar_poly.area + rotated_poly.area - inter
    return inter / max(union, 1e-9)
# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _obb_to_aabb_xy(boxes):
    """
    Convert oriented boxes on the xy plane to their circumscribed AABB.
    boxes: (K, 7) with columns (cx, cy, cz, dx, dy, dz, heading)
    """
    cx, cy = boxes[:, 0], boxes[:, 1]
    dx, dy = boxes[:, 3], boxes[:, 4]
    cos_h = np.abs(np.cos(boxes[:, 6]))
    sin_h = np.abs(np.sin(boxes[:, 6]))
    half_x = 0.5 * (dx * cos_h + dy * sin_h)
    half_y = 0.5 * (dx * sin_h + dy * cos_h)
    return cx - half_x, cy - half_y, cx + half_x, cy + half_y


def _pillar_box_iou_xy(pillar_bounds, box_aabb):
    """
    AABB-AABB 2D IoU between pillar and box AABB. With beta=1e-3 the
    distinction vs exact rotated-box IoU is essentially nil.
    """
    px_min, py_min, px_max, py_max = pillar_bounds
    bx_min, by_min, bx_max, by_max = box_aabb
    iw = max(0.0, min(px_max, bx_max) - max(px_min, bx_min))
    ih = max(0.0, min(py_max, by_max) - max(py_min, by_min))
    inter = iw * ih
    pillar_area = (px_max - px_min) * (py_max - py_min)
    box_area = max(1e-9, (bx_max - bx_min) * (by_max - by_min))
    union = pillar_area + box_area - inter
    return inter / max(1e-9, union)


# ---------------------------------------------------------------------------
# LOPDefense
# ---------------------------------------------------------------------------

class LOPDefense:
    """
    Post-hoc LOP filter around any 3D object detector.

    Args:
        ckpt_path:           Path to a checkpoint produced by train_lop.py.
        pc_range:            [xmin, ymin, zmin, xmax, ymax, zmax] from training.
        pillar_size:         Must match training. Default 1.0 m.
        num_points:          M_pc from paper. Must match training. Default 1024.
        boundary:            Object-level threshold B. Box kept iff
                             ratio_of_positive_pillars > boundary.
                             Paper sweeps {0.5, 0.6}. Default 0.5.
        pillar_thresh:       Per-pillar binarization threshold. Use the
                             best_thresh from training (e.g. 0.47), NOT 0.5.
        pillar_iou_beta:     Min pillar/box 2D IoU to vote. Paper: 1e-3.
        class_filter:        List of class labels to apply defense to.
                             None = apply to all (legacy v1 behavior).
                             [1] = only Vehicle (recommended for vehicle-trained
                             LOP).
        keep_if_no_pillars:  If a box has no pillars passing the IoU filter,
                             keep it (True) or eliminate it (False, paper).
        seed:                Subsampling seed for deterministic inference.
        device:              'cuda' or 'cpu'.
    """

    #boundary B: object level threshold for keeping vs eliminating a predicted box. 
    # for B = 0.5--> need 50% of pillars to look real .
    # sweep [0.4, 0.5, 0.6]
    def __init__(
        self,
        ckpt_path,
        pc_range,
        pillar_size=1.0,
        num_points=1024,
        input_dim=7,
        boundary=0.5,
        pillar_thresh=0.5,
        pillar_iou_beta=1e-3,
        class_filter=None,
        keep_if_no_pillars=False,
        seed=0,
        device=None,
    ):
        self.pc_range = np.asarray(pc_range, dtype=np.float32)
        self.pillar_size = pillar_size
        self.num_points = num_points
        self.input_dim = input_dim
        self.boundary = boundary
        self.pillar_thresh = pillar_thresh
        self.beta = pillar_iou_beta
        self.class_filter = (
            None if class_filter is None
            else np.asarray(class_filter, dtype=np.int64)
        )
        self.keep_if_no_pillars = keep_if_no_pillars
        self._rng = np.random.default_rng(seed)

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = PointNetLOP(input_dim=input_dim, num_points=num_points)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Pillar collection
    # ------------------------------------------------------------------

    def _collect_relevant_pillars(self, points, pred_boxes):
        xmin, ymin, _, xmax, ymax, _ = self.pc_range
        ps = self.pillar_size
        nx = int(np.ceil((xmax - xmin) / ps))
        ny = int(np.ceil((ymax - ymin) / ps))

        in_range = (
            (points[:, 0] >= xmin) & (points[:, 0] < xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] < ymax)
        )
        pts = np.asarray(points[in_range, :4], dtype=np.float32)

        px = np.floor((pts[:, 0] - xmin) / ps).astype(np.int32)
        py = np.floor((pts[:, 1] - ymin) / ps).astype(np.int32)
        pid = px * ny + py

        box_x0, box_y0, box_x1, box_y1 = _obb_to_aabb_xy(pred_boxes)
        cell_x0 = np.clip(np.floor((box_x0 - xmin) / ps).astype(np.int32), 0, nx - 1)
        cell_x1 = np.clip(np.floor((box_x1 - xmin) / ps).astype(np.int32), 0, nx - 1)
        cell_y0 = np.clip(np.floor((box_y0 - ymin) / ps).astype(np.int32), 0, ny - 1)
        cell_y1 = np.clip(np.floor((box_y1 - ymin) / ps).astype(np.int32), 0, ny - 1)
        
        box_polys = [
            _rotated_box_to_polygon(
                pred_boxes[k, 0],  # cx
                pred_boxes[k, 1],  # cy
                pred_boxes[k, 3],  # dx
                pred_boxes[k, 4],  # dy
                pred_boxes[k, 6],  # heading
            )
            for k in range(len(pred_boxes))
        ]
        

        pid_to_feat_idx = {}
        box_pillar_ids = [[] for _ in range(len(pred_boxes))]
        feats_list = []

        order = np.argsort(pid, kind="stable")
        sorted_pid = pid[order]
        sorted_pts = pts[order]

        if len(sorted_pid) > 0:
            change = np.concatenate([[True], sorted_pid[1:] != sorted_pid[:-1]])
            starts = np.flatnonzero(change)
            ends = np.concatenate([starts[1:], [len(sorted_pid)]])
            pid_slices = {
                int(sorted_pid[s]): (int(s), int(e))
                for s, e in zip(starts, ends)
            }
        else:
            pid_slices = {}

        for k in range(len(pred_boxes)):
            rotated_poly = box_polys[k]
            for cx in range(cell_x0[k], cell_x1[k] + 1):
                for cy in range(cell_y0[k], cell_y1[k] + 1):
                    this_pid = int(cx * ny + cy)
                    if this_pid not in pid_slices:
                        continue
                    pillar_xmin = xmin + cx * ps
                    pillar_ymin = ymin + cy * ps
                    pillar_bounds = (
                        pillar_xmin, pillar_ymin,
                        pillar_xmin + ps, pillar_ymin + ps,
                    )
                    # Exact rotated-box IoU instead of AABB approximation,
                    # to match training-time pillar labeling semantics.
                    if _pillar_rotated_box_iou(pillar_bounds, rotated_poly) < self.beta:
                        continue

                    if this_pid not in pid_to_feat_idx:
                        s, e = pid_slices[this_pid]
                        pil_pts = sorted_pts[s:e]
                        feats = self._build_pillar_feat(
                            pil_pts,
                            center_x=pillar_xmin + 0.5 * ps,
                            center_y=pillar_ymin + 0.5 * ps,
                        )
                        pid_to_feat_idx[this_pid] = len(feats_list)
                        feats_list.append(feats)
                    box_pillar_ids[k].append(pid_to_feat_idx[this_pid])

        if len(feats_list) == 0:
            return None, box_pillar_ids

        pillar_feats = np.stack(feats_list, axis=0)
        pillar_feats = torch.from_numpy(pillar_feats).to(self.device)
        return pillar_feats, box_pillar_ids

    def _build_pillar_feat(self, pts, center_x, center_y):
        n_in = len(pts)
        if n_in > self.num_points:
            idx = self._rng.choice(n_in, self.num_points, replace=False)
            pts = pts[idx]
            n_used = self.num_points
        else:
            n_used = n_in

        feats = np.zeros((self.num_points, self.input_dim), dtype=np.float32)
        x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        feats[:n_used, 0] = x - center_x
        feats[:n_used, 1] = y - center_y
        feats[:n_used, 2] = x
        feats[:n_used, 3] = y
        feats[:n_used, 4] = z
        feats[:n_used, 5] = intensity
        feats[:n_used, 6] = np.sqrt(x * x + y * y + z * z)
        return feats

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def filter(self, points, pred_boxes, pred_labels=None):
        """
        Args:
            points:       (N_pts, 4+) raw lidar points (x, y, z, intensity, ...).
            pred_boxes:   (K, 7) (cx, cy, cz, dx, dy, dz, heading).
            pred_labels:  (K,) integer class labels. Required if class_filter
                          was set in __init__. Vehicle/Pedestrian/Cyclist
                          conventions: 1 = Vehicle, 2 = Pedestrian, 3 = Cyclist
                          for OpenPCDet/Waymo.

        Returns:
            keep_mask:  (K,) bool. True = keep, False = eliminate.
        """
        points = _to_numpy(points).astype(np.float32)
        pred_boxes = _to_numpy(pred_boxes).astype(np.float32)
        if pred_labels is not None:
            pred_labels = _to_numpy(pred_labels).astype(np.int64)
        K = len(pred_boxes)

        if K == 0:
            return np.ones((0,), dtype=bool)

        # By default keep everything. We only flip to False for boxes that are
        # (a) in the class_filter set AND (b) fail the LOP test.
        keep = np.ones(K, dtype=bool)

        # Determine which boxes the defense should evaluate.
        if self.class_filter is None:
            evaluate = np.ones(K, dtype=bool)
        else:
            assert pred_labels is not None, (
                "class_filter is set but pred_labels was not provided to filter()."
            )
            pred_labels = np.asarray(pred_labels, dtype=np.int64)
            evaluate = np.isin(pred_labels, self.class_filter)

        if not evaluate.any():
            return keep  # nothing for the LOP to do

        eval_indices = np.flatnonzero(evaluate)
        eval_boxes = pred_boxes[eval_indices]

        pillar_feats, box_pillar_ids = self._collect_relevant_pillars(
            points, eval_boxes,
        )

        if pillar_feats is None:
            # Degenerate frame: no qualifying pillars at all.
            for i, k in enumerate(eval_indices):
                keep[k] = self.keep_if_no_pillars
            return keep

        logits = self.model(pillar_feats)
        scores01 = (torch.sigmoid(logits) > self.pillar_thresh).int().cpu().numpy()

        for i, k in enumerate(eval_indices):
            pids = box_pillar_ids[i]
            if len(pids) == 0:
                keep[k] = self.keep_if_no_pillars
                continue
            ratio = scores01[pids].mean()
            keep[k] = ratio > self.boundary

        return keep
