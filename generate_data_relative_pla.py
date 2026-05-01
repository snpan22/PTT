#!/usr/bin/env python
"""
generate_data_relative_pla.py

PLA-LiDAR constrained variant of generate_data_relative.py. Identical
pipeline with three additional physical constraints applied to every
spoofed frame (current and historical) before raycasting:

  1. Attacker-centric angular cone   : spoof points must lie within
                                       +/-PLA_CONE_HALF_AZ_DEG azimuth and
                                       +/-PLA_CONE_HALF_EL_DEG elevation of
                                       the attacker direction in the lag-i
                                       ego frame.
  2. Off-axis range noise            : per-point additive noise along the
                                       ray direction, with sigma growing
                                       linearly in the point's off-axis
                                       angle from the attacker axis
                                       (calibrated to PLA-LiDAR Fig 9c).
  3. Per-frame azimuthal jitter      : one scalar rotation about z with
                                       std PLA_JITTER_STD_DEG applied to
                                       all spoof points in the frame.

Threat model (relative-fixed). The phantom is kept at a constant ego-
relative position across the 32-frame spoof run. This simulates the
moving-vehicle scenario from PLA-LiDAR Section VI-C (Fig 19), where the
attacker car travels alongside the victim car at a matched speed and the
attacker is 5-15 m from the victim LiDAR. Because the attacker moves with
the ego, its ego-relative position is also constant: phantom and attacker
both stay at fixed ego-frame coordinates for the whole run.

Attacker placement (relative-fixed). Deterministic offset from phantom
centre, in ego-aligned axes:
  - longitudinal  : +3.0 m forward of phantom centre
  - lateral       : +/-(0.5*phantom_width + 1.1 m), sign matches the
                    phantom's lateral side so the attacker sits further
                    outboard than the phantom
  - vertical      : -0.3 m (roughly ground level below phantom centroid)

This same ego-relative attacker position is reused unchanged at every
lag frame inside process_4frames -- no inv(pose_lag_i) transformation
needed, unlike in the global variant.

Distance regime. Phantoms are placed 15-20 m forward and stay there
(ego-relative) for all 32 frames. No ego drift away from the phantom.
Every frame in the run sees the phantom at the same 15-20 m range, so
the attacker-difficulty regime is uniform across the dataset.

Per-frame PLA retention counts land in the saved dataset dicts. Each
spoof run's attacker offset and ego-relative position are recorded in
the spoof_annotation; an aggregate summary is written to each segment's
annotations pkl.
"""
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

elevations = np.load('waymo_top_lidar_inclinations.npy')
stats = np.load("range_conditioned_stats.npy", allow_pickle=True).item()


# ---------------------------------------------------------------------------
# PLA-LiDAR physical constraint parameters
# ---------------------------------------------------------------------------
# Cone half-angles. PLA-LiDAR reports ~30 deg horizontal x ~30 deg vertical
# total coverage for the 'wall' injection mode; we use +/-15 deg each side.
PLA_CONE_HALF_AZ_DEG = 10.0
PLA_CONE_HALF_EL_DEG = 15.0

# Per-frame azimuthal jitter (rotation about z) standard deviation.
PLA_JITTER_STD_DEG = 0.2

# Off-axis range-noise model. sigma_r(phi_deg) = base + slope * |phi_deg|.
# Calibrated so that ~90% of points within |phi| <= 6 deg have range error
# below ~0.102 m, widening to roughly ~0.25 m by |phi| <= 15 deg
# (consistent with PLA-LiDAR Fig 9c).
PLA_RANGE_NOISE_BASE_M = 0.05
PLA_RANGE_NOISE_SLOPE_M_PER_DEG = 0.0067

# ---------------------------------------------------------------------------
# Attacker offset (deterministic, ego-relative, constant for a spoof run)
# ---------------------------------------------------------------------------
# In relative_position_fixed the attacker's ego-relative position does not
# change across lag frames, so there is no per-lag sampling. The offset
# is deterministic per run, in ego-aligned axes:
#   x = forward (+3.0 m ahead of phantom centre)
#   y = +/-(0.5 * phantom_width + clearance) with sign matching the
#         phantom's side of the ego (keeps attacker further outboard than
#         the phantom rather than between phantom and ego)
#   z = -0.3 m (roughly ground level below phantom centroid)
PLA_ATTACKER_LON_OFFSET_M = 3.0
PLA_ATTACKER_LATERAL_CLEARANCE_M = 1.1
PLA_ATTACKER_Z_OFFSET_M = -0.3


def safe_ratio(a, b):
    return a / b if b > 0 else 0.0


# ---------------------------------------------------------------------------
# PLA constraint function
# ---------------------------------------------------------------------------
def pla_constrain(spoof_pts_ego,
                  attacker_pos_ego,
                  rng=None,
                  cone_half_az_deg=PLA_CONE_HALF_AZ_DEG,
                  cone_half_el_deg=PLA_CONE_HALF_EL_DEG,
                  jitter_std_deg=PLA_JITTER_STD_DEG,
                  range_noise_base=PLA_RANGE_NOISE_BASE_M,
                  range_noise_slope=PLA_RANGE_NOISE_SLOPE_M_PER_DEG):
    """
    Apply PLA-LiDAR physical injection constraints to spoof points.

    Parameters
    ----------
    spoof_pts_ego : (N, D>=3) array
        Spoof points in the target ego frame. First three columns are
        xyz; remaining columns (intensity, elongation, ...) are carried
        through the cone mask untouched.
    attacker_pos_ego : (3,) array-like
        Attacker position in the same ego frame. 
    rng : np.random.Generator or None
        If None, uses np.random.default_rng().

    Returns
    -------
    spoof_pts_constrained : (M, D) array with M <= N
    stats : dict with per-call retention counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_in = int(spoof_pts_ego.shape[0])
    stats = {
        'n_requested': n_in,
        'n_after_cone': 0,
        'n_after_pla': 0,
    }
    if n_in == 0:
        return spoof_pts_ego, stats

    attacker_pos_ego = np.asarray(attacker_pos_ego, dtype=np.float64).reshape(3)

    xyz = spoof_pts_ego[:, :3].astype(np.float64, copy=True)

    # ---- 1. Attacker-centric cone mask -------------------------------------
    att_az = np.arctan2(attacker_pos_ego[1], attacker_pos_ego[0])
    att_rxy = np.hypot(attacker_pos_ego[0], attacker_pos_ego[1])
    # clamp to avoid el=undefined at exactly ego origin
    att_rxy_safe = max(att_rxy, 1e-6)
    att_el = np.arctan2(attacker_pos_ego[2], att_rxy_safe)

    pt_az = np.arctan2(xyz[:, 1], xyz[:, 0])
    pt_rxy = np.hypot(xyz[:, 0], xyz[:, 1])
    pt_rxy_safe = np.where(pt_rxy > 1e-6, pt_rxy, 1e-6)
    pt_el = np.arctan2(xyz[:, 2], pt_rxy_safe)

    # Azimuth difference wrapped to (-pi, pi]
    d_az = np.mod(pt_az - att_az + np.pi, 2 * np.pi) - np.pi
    d_el = pt_el - att_el

    cone_mask = (np.abs(d_az) <= np.deg2rad(cone_half_az_deg)) & \
                (np.abs(d_el) <= np.deg2rad(cone_half_el_deg))

    spoof_after_cone = spoof_pts_ego[cone_mask]
    d_az_kept = d_az[cone_mask]
    d_el_kept = d_el[cone_mask]
    n_after_cone = int(spoof_after_cone.shape[0])
    stats['n_after_cone'] = n_after_cone

    if n_after_cone == 0:
        stats['n_after_pla'] = 0
        return spoof_after_cone, stats

    # Make a writable copy for noise/jitter so we don't mutate caller data.
    out = spoof_after_cone.copy()
    xyz_kept = out[:, :3].astype(np.float64, copy=False)

    # ---- 2. Off-axis range noise ------------------------------------------
    off_axis_deg = np.rad2deg(np.sqrt(d_az_kept ** 2 + d_el_kept ** 2))
    sigma_r = range_noise_base + range_noise_slope * off_axis_deg
    delta_r = rng.normal(0.0, sigma_r)

    r_pt = np.linalg.norm(xyz_kept, axis=1)
    r_pt_safe = np.where(r_pt > 1e-6, r_pt, 1e-6)
    ray_unit = xyz_kept / r_pt_safe[:, None]
    xyz_kept = xyz_kept + ray_unit * delta_r[:, None]

    # ---- 3. Per-frame azimuthal jitter ------------------------------------
    jitter_rad = float(np.deg2rad(rng.normal(0.0, jitter_std_deg)))
    cj = np.cos(jitter_rad)
    sj = np.sin(jitter_rad)
    Rz = np.array([[cj, -sj, 0.0],
                   [sj,  cj, 0.0],
                   [0.0, 0.0, 1.0]])
    xyz_kept = xyz_kept @ Rz.T

    # Write xyz back, preserving intensity/elongation and original dtype.
    out[:, :3] = xyz_kept.astype(out.dtype, copy=False)
    stats['n_after_pla'] = int(out.shape[0])
    return out, stats


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
    parser.add_argument("--cp_mf_ckpt", required=True)

    parser.add_argument("--cp_sf_cfg", required=True)
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to runtime log file"
    )
    parser.add_argument("--save_dir_dataset", required=True)
    parser.add_argument("--save_dir_annos", required=True)
    parser.add_argument("--trace_file", required=True)

    return parser.parse_args()


def center_trace(selected_trace):
    pts = selected_trace['points'][:, :2]
    box = selected_trace['box']

    centered_points_trace = pts - box[:2]

    # original yaw
    yaw = box[6]

    # rotate points so object becomes ego-aligned
    angle = -yaw
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([[c, -s],
                  [s,  c]])

    rotated_trace = centered_points_trace @ R.T

    box_new = box.copy()
    box_new[6] = 0.0
    box_new[0] = 0.0
    box_new[1] = 0.0

    return rotated_trace, box_new


def place_trace(traces, dataset_mf, perturbation, window_seg_min, window_seg_max):
    """
    For relative_position_fixed:
      The trace is placed in ego-relative coords at the (implicit) anchor
      frame and reused as-is at every lag frame inside process_4frames.
      This models an attacker that moves with the ego at the same speed,
      so the phantom stays at a constant ego-relative position for the
      whole 32-frame window. There is NO anchor_pose argument and no
      world-frame transformation -- this is the structural difference
      from the global variant.

    Attacker placement (relative-fixed). Because the attacker is also
    moving with the ego in this regime, its ego-relative position is
    constant too. We place it at a deterministic offset from the phantom
    centre in ego-aligned axes (+3.0 m forward, +/-(0.5 * width + 1.1) m
    lateral with sign matching the phantom's side of the ego, -0.3 m
    vertical). This same ego-relative position is stored in
    target_frames_attacker and consumed unchanged by pla_constrain at
    every lag frame -- no pose transformation needed.
    """
    rng = np.random.default_rng()
    trace = rng.choice(traces)
    relative_position_offset = np.array([rng.uniform(15, 20), rng.uniform(-3, 3)])

    centered_trace, centered_box = center_trace(trace)

    # --- random small yaw ---
    yaw_new = rng.uniform(-0.2, 0.2)
    c = np.cos(yaw_new)
    s = np.sin(yaw_new)
    R = np.array([[c, -s],
                  [s,  c]])
    centered_trace = centered_trace @ R.T
    centered_box[6] = float(yaw_new)
    # ------------------------

    # place in ego-relative coords of anchor frame
    front_near_trace_planar = centered_trace + relative_position_offset
    front_near_trace = np.concatenate(
        [front_near_trace_planar, trace['points'][:, 2].reshape(-1, 1)], axis=1
    )

    # spoof_points stay in ego-relative coords: relative-fixed means they
    # are reused unchanged at every lag frame. This is the key behavioural
    # difference from the global variant, where spoof_points get rotated
    # and translated into world coords here.
    spoof_points = np.hstack((front_near_trace, trace['points'][:, 3:5]))

    front_near_box = centered_box.copy()
    front_near_box[0] += relative_position_offset[0]
    front_near_box[1] += relative_position_offset[1]

    # Phantom centre in ego-relative (anchor) coords.
    phantom_xyz_anchor_ego = np.array([front_near_box[0],
                                       front_near_box[1],
                                       front_near_box[2]], dtype=np.float64)

    # --- Deterministic attacker placement in ego-relative coords ----------
    # Lateral offset uses the phantom's width (dy in object frame after
    # the centering yaw rotation) plus a fixed clearance. Sign matches
    # phantom side so the attacker ends up further outboard than the
    # phantom.
    phantom_width = centered_box[4]  # [x, y, z, dx, dy(=width), dz, yaw]
    sign = 1.0 if phantom_xyz_anchor_ego[1] >= 0 else -1.0
    lateral_offset = 0.5 * phantom_width + PLA_ATTACKER_LATERAL_CLEARANCE_M

    attacker_offset_anchor_ego = np.array([
        PLA_ATTACKER_LON_OFFSET_M,
        sign * lateral_offset,
        PLA_ATTACKER_Z_OFFSET_M,
    ], dtype=np.float64)
    attacker_xyz_anchor_ego = phantom_xyz_anchor_ego + attacker_offset_anchor_ego

    annotation = {
        'window': [window_seg_min, window_seg_max],
        'perturbation': perturbation,
        # attacker bookkeeping (ego-relative; constant across the run)
        'attacker_offset_anchor_ego': attacker_offset_anchor_ego.tolist(),
        'attacker_pos_ego': attacker_xyz_anchor_ego.tolist(),
        'phantom_pos_anchor_ego': phantom_xyz_anchor_ego.tolist(),
    }

    return spoof_points, annotation, front_near_box, attacker_xyz_anchor_ego


def isolate_frame_points(pts, lag):
    pts_clean = np.array(pts, copy=True)
    lags = np.round(pts_clean[:, -1], decimals=1)
    target_lag = 0.1 * lag
    mask = (lags == target_lag)
    current_frame = pts_clean[mask]
    return current_frame


def ground_anchor_spoof(points, spoof, gt, radius=2.0):
    """
    points : scene lidar points (NxD)
    spf_cur : spoof points (Mx5 or Mx3 depending on pipeline)

    returns: ground anchored spoof points
    """
    gt_box = gt.copy()
    spf_cur = spoof.copy()

    scene_xyz = points[:, :3]

    spoof_xy = spf_cur[:, :2]
    center_xy = spoof_xy.mean(axis=0)

    dists = np.linalg.norm(scene_xyz[:, :2] - center_xy, axis=1)
    local_mask = dists < radius

    if np.sum(local_mask) > 20:
        local_points = scene_xyz[local_mask]

        ground_z = np.percentile(local_points[:, 2], 5)
        spoof_bottom_z = np.min(spf_cur[:, 2])

        dz = ground_z - spoof_bottom_z

        spf_cur[:, 2] += dz
        gt_box[2] += dz

    return spf_cur, gt_box


def spoof_frame(spoof_points, points, gt_box):
    global elevations, stats
    # --- Ground anchoring correction ---
    n_spoof_r = spoof_points.shape[0]
    spf_cur, gt_box = ground_anchor_spoof(points, spoof_points, gt_box)

    # empirically sample intensities and elongations
    r_avg = helpers_ptt.avg_spoof_range(spf_cur)
    band_stats = helpers_ptt.get_stats_for_range(r_avg, stats)
    P_two = 0.01
    if (band_stats is not None):
        I = np.random.lognormal(
                band_stats["log_mu"],
                band_stats["log_sigma"],
                size=spf_cur.shape[0]
            )
        I = np.clip(I, band_stats["intensity_p1"], band_stats["intensity_p99"])
        E = np.random.choice(band_stats["elong_pool"], size=spf_cur.shape[0])
        spf_cur[:, 3] = I
        spf_cur[:, 4] = E

    dtheta = np.deg2rad(0.1358)

    # --- Convert to spherical ---
    r_trace, theta_trace, phi_trace = helpers_ptt.cart_2_spherical(spf_cur)
    scene_xyz = points[:, :3]
    r_scene, theta_scene, phi_scene = helpers_ptt.cart_2_spherical(scene_xyz)

    # --- Discretize scene points ---
    az_scene = np.mod(theta_scene, 2*np.pi)
    az_idx_scene = np.floor(az_scene / dtheta).astype(np.int32)
    beam_idx_scene = np.abs(elevations[:, None] - phi_scene).argmin(axis=0)

    # --- Sort scene by (az_idx, beam_idx, range) ---
    order = np.lexsort((r_scene, beam_idx_scene, az_idx_scene))
    az_idx_scene = az_idx_scene[order]
    beam_idx_scene = beam_idx_scene[order]
    r_scene = r_scene[order]
    scene_indices_sorted = order

    # Identify group boundaries
    keys = az_idx_scene * 1000 + beam_idx_scene
    unique_keys, group_start = np.unique(keys, return_index=True)
    group_end = np.r_[group_start[1:], len(keys)]

    key_to_slice = {
        k: (start, end)
        for k, start, end in zip(unique_keys, group_start, group_end)
    }

    remove_spoof = []
    remove_scene = []

    # --- Process spoof points ---
    for idx, (r, theta, phi) in enumerate(zip(r_trace, theta_trace, phi_trace)):
        az = theta % (2*np.pi)
        az_idx = int(np.floor(az / dtheta))
        beam_idx = np.argmin(np.abs(elevations - phi))
        key = az_idx * 1000 + beam_idx

        if key not in key_to_slice:
            continue

        start, end = key_to_slice[key]
        r_group = r_scene[start:end]
        idx_group = scene_indices_sorted[start:end]

        if len(r_group) == 0:
            continue

        # Case 1: spoof behind first real return
        if r_group[0] < r:
            remove_spoof.append(idx)

        # Case 2: spoof in front of all
        elif r < r_group[0]:
            if np.random.rand() < P_two:
                if len(idx_group) > 0:
                    remove_scene.extend(idx_group[1:].tolist())
            else:
                remove_scene.extend(idx_group.tolist())

        # Case 3: spoof between returns
        else:
            k = np.searchsorted(r_group, r, side='right')

            if k < len(r_group):
                if np.random.rand() < P_two:
                    remove_scene.extend(idx_group[k+1:].tolist())
                else:
                    remove_scene.extend(idx_group[k:].tolist())

    # --- Apply removals ---
    mask_spoof = np.ones(spf_cur.shape[0], dtype=bool)
    mask_spoof[remove_spoof] = False
    spoof_points_rc = spf_cur[mask_spoof]

    removed_unique = len(set(remove_scene))
    kept_spoof = spoof_points_rc.shape[0]

    mask_scene = np.ones(points.shape[0], dtype=bool)
    mask_scene[remove_scene] = False
    scene_points_rc = points[mask_scene]

    # Preserve lag column exactly as before
    lag_value = points[0, -1]
    lag_col = np.full((spoof_points_rc.shape[0], 1), lag_value, dtype=np.float32)
    spf_pts_rc_lag = np.hstack([spoof_points_rc, lag_col])

    frame_points_spoof_rc = np.concatenate([scene_points_rc, spf_pts_rc_lag], axis=0)

    return frame_points_spoof_rc, gt_box, n_spoof_r, kept_spoof


def process_4frames(dataset_mf,
                    dataset_sf,
                    frame_idx,
                    target_frames,
                    frame_seq,
                    perturbation,
                    mf_to_sf_global,
                    target_frames_anno,
                    target_frames_attacker,
                    pla_rng=None):

    if pla_rng is None:
        pla_rng = np.random.default_rng()

    target_frames_copy = {k: v.copy() if v is not None else None for k, v in target_frames.items()}
    target_frames_anno_copy = {k: v.copy() if v is not None else None for k, v in target_frames_anno.items()}
    # attacker map values are (3,) ego-frame positions (constant across lag
    # frames of a given spoof run in relative mode -- this is the key
    # difference from the global variant where they are world-frame and
    # need to be transformed per lag).
    target_frames_attacker_copy = {k: (np.asarray(v, dtype=np.float64).copy()
                                       if v is not None else None)
                                   for k, v in target_frames_attacker.items()}

    lag0_n_requested = 0
    lag0_n_after_cone = 0
    lag0_n_after_pla = 0
    lag0_n_after_raycast = 0
    points_frame = None
    none_spoofed = 1
    infos = dataset_mf.infos
    info_cur = infos[frame_idx]
    seq_name = info_cur['point_cloud']['lidar_sequence']
    pose_cur = info_cur['pose'].reshape(4, 4)

    n_spoof_r = 0
    n_spoof_k = 0
    gt_box_lag0 = None

    # Aggregate PLA stats across all spoofed lag frames processed in this call.
    pla_agg = {
        'n_requested': 0,
        'n_after_cone': 0,
        'n_after_pla': 0,
        'n_after_raycast': 0,
        'n_spoofed_lag_frames': 0,
        'n_spoofed_lag_frames_fully_masked': 0,
        # phantom-to-attacker angular separation at lag 0 (ego frame),
        # deg. Only set when lag 0 is a spoofed frame; otherwise None.
        'lag0_phantom_to_attacker_angle_deg': None,
    }

    indices = []
    for i in range(4):
        target_idx = frame_idx - i

        if target_idx < 0:
            target_idx = 0

        if infos[target_idx]['point_cloud']['lidar_sequence'] != seq_name:
            target_idx = indices[-1] if indices else frame_idx

        indices.append(target_idx)

    has_spoof = any(target_frames_copy[i] is not None for i in indices)
    if not has_spoof:
        pts = dataset_sf[mf_to_sf_global[frame_idx]]['points']
        points_frame = np.hstack([pts, np.zeros((pts.shape[0], 1))])
        return (dataset_mf[frame_idx]['points'], none_spoofed, points_frame,
                gt_box_lag0, n_spoof_r, n_spoof_k, pla_agg, lag0_n_requested,
                lag0_n_after_cone,
                lag0_n_after_pla,
                lag0_n_after_raycast)

    merged_points_list = []
    for i, idx in enumerate(indices):

        pts = dataset_sf[mf_to_sf_global[idx]]['points'].copy()
        points = np.hstack([pts, np.zeros((pts.shape[0], 1))])

        points[:, -1] = (frame_idx - idx) * 0.1
        pose_lag_i_frame = infos[idx]['pose'].reshape(4, 4)

        if (target_frames_copy[idx] is not None):

            points_lag_i_frame_in_lag_i_coords_full = points
            spoof_points = target_frames_copy[idx]

            spoof_points_homo = None
            if (perturbation == "global_position_fixed"):
                spoof_points_homo = np.hstack([spoof_points[:, 0:3], np.ones((spoof_points.shape[0], 1))])
                spf_cur = spoof_points_homo @ np.linalg.inv(pose_lag_i_frame).T

            elif (perturbation == "relative_position_fixed" or perturbation == "random"):
                spoof_points_homo = np.hstack([spoof_points[:, 0:3], np.ones((spoof_points.shape[0], 1))])
                spf_cur = spoof_points_homo

            src = spoof_points[idx] if isinstance(spoof_points, dict) else spoof_points
            spf_cur_lag_i = np.hstack([spf_cur[:, :3], src[:, 3:5]])

            # For global_position_fixed the stored box is in world coords;
            # transform it into the lag-i ego frame before passing to spoof_frame.
            gt_box_for_frame = target_frames_anno_copy[idx].copy()
            if perturbation == "global_position_fixed":
                box_global_hom = np.array([gt_box_for_frame[0],
                                           gt_box_for_frame[1],
                                           gt_box_for_frame[2], 1.0])
                box_ego = box_global_hom @ np.linalg.inv(pose_lag_i_frame).T
                gt_box_for_frame[0] = box_ego[0]
                gt_box_for_frame[1] = box_ego[1]
                gt_box_for_frame[2] = box_ego[2]
                gt_box_for_frame[6] = (gt_box_for_frame[6]
                                       - np.arctan2(pose_lag_i_frame[1, 0],
                                                    pose_lag_i_frame[0, 0]))

            # ---- Apply PLA-LiDAR physical constraints -----------------------
            # Relative mode: target_frames_attacker_copy[idx] is already
            # an ego-frame position (constant across the run). Use it
            # directly, no inv(pose_lag_i_frame) transform.
            # (In global mode this is where world-to-ego transform would
            # happen; that code path is absent here.)
            attacker_pos_ego_run = target_frames_attacker_copy[idx]
            if attacker_pos_ego_run is None:
                # Defensive fallback: should not happen because target_frames
                # and target_frames_attacker are populated in lockstep.
                attacker_pos_ego = gt_box_for_frame[:3].astype(np.float64)
            else:
                attacker_pos_ego = np.asarray(attacker_pos_ego_run,
                                              dtype=np.float64).reshape(3)

            spf_cur_lag_i_pla, pla_stats_i = pla_constrain(
                spf_cur_lag_i, attacker_pos_ego, rng=pla_rng
            )

            # If this lag is the current (lag-0) frame, record the angular
            # separation between phantom direction and attacker direction
            # from the ego at this frame. Gives a per-frame proxy for how
            # off-axis the phantom is, useful to interpret retention.
            if idx == frame_idx:
                lag0_n_requested = pla_stats_i['n_requested']
                lag0_n_after_cone = pla_stats_i['n_after_cone']
                lag0_n_after_pla = pla_stats_i['n_after_pla']
                phantom_dir = np.asarray(gt_box_for_frame[:3], dtype=np.float64)
                attacker_dir = np.asarray(attacker_pos_ego, dtype=np.float64)
                n1 = float(np.linalg.norm(phantom_dir))
                n2 = float(np.linalg.norm(attacker_dir))
                if n1 > 1e-9 and n2 > 1e-9:
                    cos_a = float(np.dot(phantom_dir, attacker_dir) / (n1 * n2))
                    cos_a = max(-1.0, min(1.0, cos_a))
                    pla_agg['lag0_phantom_to_attacker_angle_deg'] = \
                        float(np.rad2deg(np.arccos(cos_a)))

            pla_agg['n_requested'] += pla_stats_i['n_requested']
            pla_agg['n_after_cone'] += pla_stats_i['n_after_cone']
            pla_agg['n_after_pla'] += pla_stats_i['n_after_pla']
            pla_agg['n_spoofed_lag_frames'] += 1
            if pla_stats_i['n_after_pla'] == 0:
                pla_agg['n_spoofed_lag_frames_fully_masked'] += 1

            # If everything was masked out, skip raycasting for this lag frame:
            # there are no spoof points to inject, but the gt annotation
            # remains for downstream ASR bookkeeping.
            if spf_cur_lag_i_pla.shape[0] == 0:
                n_spoof_r_i = 0
                n_spoof_k_i = 0
                frame_points_spoof_rc = points_lag_i_frame_in_lag_i_coords_full
                gt_box = gt_box_for_frame
            else:
                frame_points_spoof_rc, gt_box, n_spoof_r_i, n_spoof_k_i = spoof_frame(
                    spf_cur_lag_i_pla,
                    points_lag_i_frame_in_lag_i_coords_full,
                    gt_box_for_frame,
                )
                if idx == frame_idx:
                    lag0_n_after_raycast = n_spoof_k_i
                    n_spoof_r = n_spoof_r_i
                    n_spoof_k = n_spoof_k_i


        
            pla_agg['n_after_raycast'] += n_spoof_k_i

            if (idx == frame_idx):
                points_frame = frame_points_spoof_rc
                gt_box_lag0 = gt_box.copy()

            # convert back to lag 0 coordinates
            frame_points_spoof_rc_hom = np.hstack((frame_points_spoof_rc[:, :3],
                                                   np.ones(frame_points_spoof_rc.shape[0]).reshape(-1, 1)))
            frame_points_spoof_rc_lag0_coords = frame_points_spoof_rc_hom @ pose_lag_i_frame.T @ np.linalg.inv(pose_cur).T

            points = np.hstack((frame_points_spoof_rc_lag0_coords[:, :-1],
                                frame_points_spoof_rc[:, 3:]))
        else:
            if (idx == frame_idx):
                points_frame = points
                gt_box_lag0 = target_frames_anno_copy[idx]

            points_last3cols = points[:, 3:]
            frame_points_hom = np.hstack((points[:, :3], np.ones(points.shape[0]).reshape(-1, 1)))
            frame_points_spoof_lag0_coords = frame_points_hom @ pose_lag_i_frame.T @ np.linalg.inv(pose_cur).T
            points = np.hstack((frame_points_spoof_lag0_coords[:, :-1], points_last3cols))

        merged_points_list.append(points)

    merged_points_array = np.vstack(merged_points_list)
    merged_points_array = merged_points_array.astype(np.float32)

    none_spoofed = 0

    return (merged_points_array, none_spoofed, points_frame, gt_box_lag0,
            n_spoof_r, n_spoof_k, pla_agg, lag0_n_requested,
            lag0_n_after_cone,
            lag0_n_after_pla,
            lag0_n_after_raycast)


def extract_one_runs(d):
    runs = []
    current_run = []

    for idx in sorted(d.keys()):
        if d[idx] == 1:
            current_run.append(idx)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []

    if current_run:
        runs.append(current_run)

    return runs


def generate_timing_classification(target_frames, window_size):
    classification = []

    keys = sorted(target_frames.keys())
    n_prev = window_size - 1

    for k in keys:
        cur = target_frames.get(k)

        prev_vals = [target_frames.get(k - i) for i in range(1, n_prev + 1)]

        prev_all_none = all(v is None for v in prev_vals)
        prev_any_not_none = any(v is not None for v in prev_vals)

        if cur is None and prev_all_none:
            classification.append(0)

        elif cur is not None and prev_all_none:
            classification.append(1)

        elif cur is not None and prev_any_not_none:
            classification.append(2)

        elif cur is None and prev_any_not_none:
            classification.append(3)

    return classification


# ---------------------------------------------------------------------------
# Segment-level PLA summary helpers
# ---------------------------------------------------------------------------
def _empty_segment_pla_totals():
    return {
        'n_requested': 0,
        'n_after_cone': 0,
        'n_after_pla': 0,
        'n_after_raycast': 0,
        'n_spoofed_lag_frames': 0,
        'n_spoofed_lag_frames_fully_masked': 0,
        'n_current_frames_with_spoof': 0,
        'lag0_n_requested': 0,
        'lag0_n_after_cone': 0,
        'lag0_n_after_raycast': 0,
        'lag0_frame_count': 0,
        
        # Collected list of lag-0 phantom-to-attacker angles (deg) for each
        # current frame that had a spoof. Used to summarize how off-axis the
        # phantom was throughout the segment.
        'lag0_phantom_to_attacker_angles_deg': [],
       
    }


def _accumulate_segment_pla(totals, pla_agg, frame_data):
    totals['n_requested'] += pla_agg['n_requested']
    totals['n_after_cone'] += pla_agg['n_after_cone']
    totals['n_after_pla'] += pla_agg['n_after_pla']
    totals['n_after_raycast'] += pla_agg['n_after_raycast']
    totals['n_spoofed_lag_frames'] += pla_agg['n_spoofed_lag_frames']
    totals['n_spoofed_lag_frames_fully_masked'] += pla_agg['n_spoofed_lag_frames_fully_masked']
    if pla_agg['n_spoofed_lag_frames'] > 0:
        totals['n_current_frames_with_spoof'] += 1
    if pla_agg.get('lag0_phantom_to_attacker_angle_deg') is not None:
        totals['lag0_phantom_to_attacker_angles_deg'].append(
            pla_agg['lag0_phantom_to_attacker_angle_deg'])
    if frame_data['pla_lag0_n_requested'] > 0:
        totals['lag0_n_requested'] += frame_data['pla_lag0_n_requested']
        totals['lag0_n_after_cone'] += frame_data['pla_lag0_n_after_cone']
        totals['lag0_n_after_raycast'] += frame_data['pla_lag0_n_after_raycast']
        totals['lag0_frame_count'] += 1


def _finalize_segment_pla_summary(totals):
    def safe_ratio(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    angles = list(totals.pop('lag0_phantom_to_attacker_angles_deg', []))

    summary = dict(totals)
    summary['cone_retention_ratio'] = safe_ratio(totals['n_after_cone'],
                                                 totals['n_requested'])
    summary['raycast_retention_ratio'] = safe_ratio(totals['n_after_raycast'],
                                                    totals['n_after_pla'])
    summary['overall_retention_ratio'] = safe_ratio(totals['n_after_raycast'],
                                                    totals['n_requested'])
    summary['fully_masked_lag_frame_ratio'] = safe_ratio(
        totals['n_spoofed_lag_frames_fully_masked'],
        totals['n_spoofed_lag_frames'],
    )
    summary['lag0_cone_keep'] = safe_ratio(
        totals['lag0_n_after_cone'],
        totals['lag0_n_requested']
    )

    summary['lag0_overall_keep'] = safe_ratio(
        totals['lag0_n_after_raycast'],
        totals['lag0_n_requested']
    )
    if angles:
        a = np.asarray(angles, dtype=np.float64)
        summary['phantom_to_attacker_angle_deg'] = {
            'count': int(a.size),
            'mean': float(a.mean()),
            'p50': float(np.median(a)),
            'p90': float(np.percentile(a, 90)),
            'max': float(a.max()),
        }
    else:
        summary['phantom_to_attacker_angle_deg'] = None
    summary['constraint_params'] = {
        'cone_half_az_deg': PLA_CONE_HALF_AZ_DEG,
        'cone_half_el_deg': PLA_CONE_HALF_EL_DEG,
        'jitter_std_deg': PLA_JITTER_STD_DEG,
        'range_noise_base_m': PLA_RANGE_NOISE_BASE_M,
        'range_noise_slope_m_per_deg': PLA_RANGE_NOISE_SLOPE_M_PER_DEG,
        'attacker_lon_offset_m': PLA_ATTACKER_LON_OFFSET_M,
        'attacker_lateral_clearance_m': PLA_ATTACKER_LATERAL_CLEARANCE_M,
        'attacker_z_offset_m': PLA_ATTACKER_Z_OFFSET_M,
    }
    return summary


def main():

    args = parse_args()
    log = logging.getLogger("spoof")
    log.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    log.addHandler(sh)

    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    log.info(
        "PLA constraints enabled (relative mode): cone=+/-%.1f deg (az) x "
        "+/-%.1f deg (el), jitter_std=%.2f deg, range_noise base=%.3f m "
        "slope=%.4f m/deg",
        PLA_CONE_HALF_AZ_DEG, PLA_CONE_HALF_EL_DEG,
        PLA_JITTER_STD_DEG,
        PLA_RANGE_NOISE_BASE_M, PLA_RANGE_NOISE_SLOPE_M_PER_DEG,
    )
    log.info(
        "PLA attacker offset (ego-relative, constant across run): "
        "forward x=%.2f m, lateral |y|=0.5*phantom_width + %.2f m "
        "(sign matches phantom side), vertical z=%.2f m",
        PLA_ATTACKER_LON_OFFSET_M, PLA_ATTACKER_LATERAL_CLEARANCE_M,
        PLA_ATTACKER_Z_OFFSET_M,
    )

    # load in centerpoint multiframe dataset
    unload_pcdet()
    sys.path.insert(0, args.cp_mf_root)

    from pcdet.config import cfg as cfg_cp
    from pcdet.config import cfg_from_yaml_file as cfg_from_yaml_file_cp
    from pcdet.datasets import build_dataloader as build_cp_loader
    from pcdet.models import build_network as build_cp_network
    from pcdet.models import load_data_to_gpu
    from pcdet.utils import common_utils

    sys.path.pop(0)

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
        logger=logger,
        training=False
    )

    cp_mf_model = build_cp_network(
        model_cfg=cfg_cp.MODEL,
        num_class=len(cfg_cp.CLASS_NAMES),
        dataset=dataset_mf
    )
    cp_mf_model.load_params_from_file(args.cp_mf_ckpt, logger=logger, to_cpu=False)
    cp_mf_model.cuda().eval()

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
        logger=logger,
        training=False
    )

    frame_ids = [info['frame_id'] for info in dataset_sf.infos]

    unique_segments_sf = sorted(list(set(fid.rsplit('_', 1)[0] for fid in frame_ids)))

    mf_segments = set(info['point_cloud']['lidar_sequence']
                      for info in dataset_mf.infos)
    common_segments = mf_segments.intersection(unique_segments_sf)

    segment_to_global = defaultdict(list)
    global_to_segment = {}
    seg_localidx_to_pose = {}

    for global_idx, info in enumerate(dataset_mf.infos):
        segment = info['point_cloud']['lidar_sequence']
        if segment in common_segments:
            local_idx = len(segment_to_global[segment])
            segment_to_global[segment].append(global_idx)
            global_to_segment[global_idx] = (segment, local_idx)
            seg_localidx_to_pose[(segment, local_idx)] = info['pose'].reshape((4, 4))

    segment_counts = defaultdict(int)

    for info in dataset_mf.infos:
        segment = info['point_cloud']['lidar_sequence']
        if segment in common_segments:
            segment_counts[segment] += 1

    segment_last_local_idx = {seg: count - 1 for seg, count in segment_counts.items()}

    target_frames_global = {}
    target_frames_seg = {}

    for seg, last_ind in segment_last_local_idx.items():
        toggle = False
        targets_global = {}
        targets_seg = {}
        for i in range(last_ind + 1):
            if i == 0:
                targets_seg[i] = None
                targets_global[segment_to_global[seg][i]] = None
            else:
                rem = i % 32
                if (not rem):
                    toggle = not toggle
                if toggle:
                    targets_seg[i] = 1
                    targets_global[segment_to_global[seg][i]] = 1
                else:
                    targets_seg[i] = None
                    targets_global[segment_to_global[seg][i]] = None
        target_frames_global[seg] = targets_global
        target_frames_seg[seg] = targets_seg

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

    segment_dataset = []
    segment_spoof_annos = []
    segment_pla_totals = _empty_segment_pla_totals()

    save_dir_dataset = args.save_dir_dataset
    os.makedirs(save_dir_dataset, exist_ok=True)
    save_dir_annos = args.save_dir_annos
    os.makedirs(save_dir_annos, exist_ok=True)

    completed = set(
        f[:-4] for f in os.listdir(save_dir_dataset)
    )

    traces_file = args.trace_file
    with open(traces_file, 'rb') as file:
        traces = pkl.load(file)

    target_frames = None
    current_segment = None
    target_frames_anno = None
    target_frames_attacker = None

    # One RNG for the whole run; reuse across pla_constrain calls.
    pla_rng = np.random.default_rng()

    def _emit_segment(segment_name):
        """Write out the currently-accumulated segment dataset, annos, and
        PLA summary, and log the retention rate."""
        if not segment_dataset:
            return
        segment_annos = {}
        segment_annos['segment'] = segment_name
        segment_annos['timing_ptt'] = generate_timing_classification(
            target_frames_seg[segment_name], 32)
        segment_annos['timing_msf'] = generate_timing_classification(
            target_frames_seg[segment_name], 4)
        segment_annos['spoof_annos'] = list(segment_spoof_annos)
        segment_annos['pla_summary'] = _finalize_segment_pla_summary(segment_pla_totals)

        dataset_path = f"{save_dir_dataset}/{segment_name}_d.pkl"
        anno_path = f"{save_dir_annos}/{segment_name}_a.pkl"

        with open(dataset_path, "wb") as f:
            pkl.dump(segment_dataset, f)
        with open(anno_path, "wb") as f:
            pkl.dump(segment_annos, f)

        s = segment_annos['pla_summary']
        log.info(
            "Saved: %s | PLA requested=%d cone=%d raycast=%d | "
            "cone_keep=%.3f raycast_given_pla=%.3f overall_keep=%.3f | "
            "lag0_cone_keep=%.3f lag0_overall_keep=%.3f | "
            "masked_lag_frames=%d/%d (%.3f)",
            segment_name,
            s['n_requested'], s['n_after_cone'], s['n_after_raycast'],
            s['cone_retention_ratio'],
            s['raycast_retention_ratio'],   # now correctly labeled
            s['overall_retention_ratio'],
            s['lag0_cone_keep'],
            s['lag0_overall_keep'],
            s['n_spoofed_lag_frames_fully_masked'], s['n_spoofed_lag_frames'],
            s['fully_masked_lag_frame_ratio']
        )

    try:
        for (frame, batch) in enumerate(test_loader_mf):
            segment = batch['frame_id'][0].rsplit("_", 1)[0]
            if segment not in common_segments:
                continue

            if segment in completed:
                continue

            if current_segment is None:
                current_segment = segment
                target_frames = target_frames_global[current_segment].copy()
                target_frames_anno = target_frames_global[current_segment].copy()
                target_frames_attacker = target_frames_global[current_segment].copy()

                runs_global = extract_one_runs(target_frames)
                for run in runs_global:
                    subset_g = {k: target_frames[k] for k in run}
                    subset_g_anno = {k: target_frames_anno[k] for k in run}
                    subset_g_att = {k: target_frames_attacker[k] for k in run}
                    window_min = global_to_segment[min(subset_g)][1]
                    window_max = global_to_segment[max(subset_g)][1]

                    # Relative mode: place_trace takes no anchor_pose
                    # and returns the attacker in ego-relative coords.
                    trace, spoof_annotation, box, attacker_ego = place_trace(
                        traces, dataset_mf, 'relative_position_fixed',
                        window_min, window_max
                    )

                    for key, key_a, key_t in zip(subset_g.keys(),
                                                 subset_g_anno.keys(),
                                                 subset_g_att.keys()):
                        subset_g[key] = trace
                        subset_g_anno[key_a] = box.copy()
                        subset_g_att[key_t] = attacker_ego.copy()

                    segment_spoof_annos.append(spoof_annotation)
                    target_frames.update(subset_g)
                    target_frames_anno.update(subset_g_anno)
                    target_frames_attacker.update(subset_g_att)

            if (segment != current_segment):
                _emit_segment(current_segment)

                segment_dataset = []
                segment_spoof_annos = []
                segment_pla_totals = _empty_segment_pla_totals()

                current_segment = segment
                target_frames = None
                target_frames_anno = None
                target_frames_attacker = None
                target_frames = target_frames_global[current_segment].copy()
                target_frames_anno = target_frames_global[current_segment].copy()
                target_frames_attacker = target_frames_global[current_segment].copy()

                runs_global = extract_one_runs(target_frames)
                for run in runs_global:
                    subset_g = {k: target_frames[k] for k in run}
                    subset_g_anno = {k: target_frames_anno[k] for k in run}
                    subset_g_att = {k: target_frames_attacker[k] for k in run}
                    window_min = global_to_segment[min(subset_g)][1]
                    window_max = global_to_segment[max(subset_g)][1]

                    trace, spoof_annotation, box, attacker_ego = place_trace(
                        traces, dataset_mf, 'relative_position_fixed',
                        window_min, window_max
                    )

                    for key, key_a, key_t in zip(subset_g.keys(),
                                                 subset_g_anno.keys(),
                                                 subset_g_att.keys()):
                        subset_g[key] = trace
                        subset_g_anno[key_a] = box.copy()
                        subset_g_att[key_t] = attacker_ego.copy()

                    segment_spoof_annos.append(spoof_annotation)
                    target_frames.update(subset_g)
                    target_frames_anno.update(subset_g_anno)
                    target_frames_attacker.update(subset_g_att)

            # build dataset entry for this frame
            frame_data = {}
            mf_item = dataset_mf[frame].copy()
            frame_data['frame_id'] = mf_item['frame_id']
            frame_data['pose'] = mf_item['poses'][:4]
            frame_data['gt_boxes'] = mf_item['gt_boxes']

            frame_seq = global_to_segment[frame][1]
            perturbation = 'relative_position_fixed'
            (four_frame_pts, none_spoofed, single_frame_points, gt_box,
             n_spoof_r, n_spoof_k, pla_agg,  lag0_n_requested,
             lag0_n_after_cone,
             lag0_n_after_pla,
             lag0_n_after_raycast) = process_4frames(
                dataset_mf,
                dataset_sf,
                frame,
                target_frames,
                frame_seq,
                perturbation,
                mf_to_sf_global,
                target_frames_anno,
                target_frames_attacker,
                pla_rng=pla_rng,
            )

            if (none_spoofed):
                load_data_to_gpu(batch)
                cp_mf_model.eval()
                with torch.no_grad():
                    pred, _ = cp_mf_model(batch)
            else:
                dict_mf_mod = dataset_mf[frame].copy()
                dict_mf_mod['batch_size'] = 1

                dict_mf_mod['points'] = four_frame_pts
                for k in ['voxels', 'voxel_coords', 'voxel_num_points']:
                    dict_mf_mod.pop(k, None)
                dict_mf_mod = helpers_ptt.inject_gt_names(dict_mf_mod, dataset_mf.class_names)
                dict_mf_mod = dataset_mf.prepare_data(dict_mf_mod)

                load_data_to_gpu(dict_mf_mod)
                batch_mf_mod = helpers_ptt.convert_to_batch_cp_mf(dict_mf_mod)

                cp_mf_model.eval()
                with torch.no_grad():
                    pred, _ = cp_mf_model(batch_mf_mod)

            frame_data['pred_boxes'] = pred[0]['pred_boxes']
            frame_data['pred_scores'] = pred[0]['pred_scores']
            frame_data['pred_labels'] = pred[0]['pred_labels']
            frame_data['points'] = single_frame_points
            frame_data['spoof_gt'] = gt_box
            frame_data['n_spoof_r'] = n_spoof_r
            frame_data['n_spoof_k'] = n_spoof_k

            # ---- PLA per-frame stats -----------------------------------
            # Counts aggregated across all spoofed lag frames in this
            # timestep's 4-frame input window.
            frame_data['pla_lag0_n_requested'] = lag0_n_requested
            frame_data['pla_lag0_n_after_cone'] = lag0_n_after_cone
            frame_data['pla_lag0_n_after_pla'] = lag0_n_after_pla
            frame_data['pla_lag0_n_after_raycast'] = lag0_n_after_raycast
            
            frame_data['pla_lag0_cone_keep'] = safe_ratio(
                lag0_n_after_cone, lag0_n_requested
            )

            frame_data['pla_lag0_overall_keep'] = safe_ratio(
                lag0_n_after_raycast, lag0_n_requested
            )

            frame_data['pla_n_requested'] = int(pla_agg['n_requested'])
            frame_data['pla_n_after_cone'] = int(pla_agg['n_after_cone'])
            frame_data['pla_n_after_pla'] = int(pla_agg['n_after_pla'])
            frame_data['pla_n_after_raycast'] = int(pla_agg['n_after_raycast'])
            frame_data['pla_n_spoofed_lag_frames'] = int(pla_agg['n_spoofed_lag_frames'])
            frame_data['pla_n_spoofed_lag_frames_fully_masked'] = int(
                pla_agg['n_spoofed_lag_frames_fully_masked'])
            # Phantom-to-attacker angular separation at lag 0 (deg), or None
            # if lag 0 was not a spoofed frame. Lets you filter / correlate
            # retention and ASR against how off-axis the phantom was.
            frame_data['pla_lag0_phantom_to_attacker_angle_deg'] = \
                pla_agg['lag0_phantom_to_attacker_angle_deg']

            _accumulate_segment_pla(segment_pla_totals, pla_agg, frame_data)

            segment_dataset.append(frame_data)

    except Exception:
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        raise

    if segment_dataset:
        _emit_segment(current_segment)

    return


if __name__ == "__main__":
    main()
