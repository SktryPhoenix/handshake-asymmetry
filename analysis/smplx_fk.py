#!/usr/bin/env python3
"""SMPL-H 52-joint motion loading + SMPL-X forward kinematics.

Why this file exists
--------------------
Human-interaction datasets (Inter-X, CHI3D, Hi4D / MotionHub mirrors) store axis-angle
parameters for a 52-joint SMPL-H skeleton. Most analysis needs *joint coordinates*,
so one SMPL-X forward pass is required.

Joint layout notes
------------------
SMPL-H (52 joints):  1 root + 21 body + 15 left hand + 15 right hand
SMPL-X (55 joints):  the first 52 are order-compatible; SMPL-X only appends
                     jaw + 2 eyeballs at the end. So filling the first 52 slots
                     with the SMPL-H pose and zero-padding the rest is correct.

Coordinate frame
----------------
`run_fk` returns **Y-up** coordinates (SMPL-native). Height is component 1, NOT 2.
(Component 2 is depth.) Use `to_z_up` if you need Z-up (MuJoCo / viser).
"""
from __future__ import annotations

import numpy as np
import torch


def load_motion(path) -> dict:
    """Load one .npz motion file, tolerating the different mirrors' key sets."""
    d = np.load(path, allow_pickle=True)
    out = {
        "poses": np.asarray(d["poses"], dtype=np.float32),
        "trans": np.asarray(d["trans"], dtype=np.float32),
        "betas": np.asarray(d["betas"], dtype=np.float32).reshape(-1),
        "raw_betas": (np.asarray(d["raw_betas"], dtype=np.float32).reshape(-1)
                      if "raw_betas" in d.files else None),
        "n_frames": (int(np.asarray(d["num_frames"]).reshape(-1)[0]) if "num_frames" in d.files
                     else int(np.asarray(d["poses"]).shape[0]),
                     )[0],
        "fps": (float(np.asarray(d["mocap_framerate"]).reshape(-1)[0])
                if "mocap_framerate" in d.files else 30.0),
    }
    if "zero_betas" in d.files:
        out["zero_betas"] = bool(np.asarray(d["zero_betas"]).reshape(-1)[0])
    if "conversion_y_shift" in d.files:
        out["y_shift"] = float(np.asarray(d["conversion_y_shift"]).reshape(-1)[0])
    return out


_MODEL_CACHE: dict = {}
NB = 100          # SMPL-X shapedirs expect 100 (300 shape + 100 expression dims)
N_EXPRESSION = 10


def _model(model_path: str):
    """Cached per process: smplx.create reloads a ~100 MB npz on every call."""
    if model_path not in _MODEL_CACHE:
        import smplx
        _MODEL_CACHE[model_path] = smplx.create(
            model_path, model_type="smplx", gender="neutral",
            num_betas=NB, num_expression_coeffs=N_EXPRESSION, use_pca=False)
    return _MODEL_CACHE[model_path]


def run_fk(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray,
           model_path: str, chunk: int = 512) -> np.ndarray:
    """Return (T, 55, 3) world joint positions in metres (Y-up)."""
    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError("expected (T,156) SMPL-H axis-angle poses, got %s" % (poses.shape,))
    model = _model(model_path)
    b = np.zeros(NB, dtype=np.float32)
    src = np.asarray(betas, np.float32).reshape(-1)
    b[:min(len(src), NB)] = src[:NB]
    n = poses.shape[0]
    out = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            k = min(chunk, n - i)
            p = torch.from_numpy(poses[i:i + k])
            out.append(model(
                global_orient=p[:, 0:3].contiguous(),
                body_pose=p[:, 3:66].contiguous(),
                left_hand_pose=p[:, 66:111].contiguous(),
                right_hand_pose=p[:, 111:156].contiguous(),
                # the 3 face joints are absent from SMPL-H; they must be zeroed
                # explicitly, otherwise smplx falls back to batch_size=1 and the
                # tensor sizes clash.
                jaw_pose=torch.zeros(k, 3), leye_pose=torch.zeros(k, 3), reye_pose=torch.zeros(k, 3),
                expression=torch.zeros(k, N_EXPRESSION),
                betas=torch.from_numpy(np.tile(b.reshape(1, -1), (k, 1))),
                transl=torch.from_numpy(trans[i:i + k]),
            ).joints.numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def to_z_up(j: np.ndarray) -> np.ndarray:
    """Y-up (x, y, z) -> Z-up (x, -z, y)."""
    return np.stack([j[..., 0], -j[..., 2], j[..., 1]], axis=-1)


# SMPL-X joint indices used throughout this repository
JOINT = {
    "pelvis": 0, "left_ankle": 7, "right_ankle": 8, "left_foot": 10, "right_foot": 11,
    "head": 15, "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
    "left_middle_mcp": 28, "right_middle_mcp": 43,
}
FEET = [JOINT["left_ankle"], JOINT["right_ankle"], JOINT["left_foot"], JOINT["right_foot"]]


def palms(j: np.ndarray) -> dict:
    """Palm centre estimated as wrist + 0.5 * (middle-MCP - wrist).

    Using the wrist alone underestimates reach and makes contact distances wrong;
    the mid-point of wrist and middle metacarpal is a good proxy for palm centre.
    """
    out = {}
    for side, pre in (("L", "left"), ("R", "right")):
        wrist = j[:, JOINT[pre + "_wrist"]]
        mcp = j[:, JOINT[pre + "_middle_mcp"]]
        out[side] = wrist + 0.5 * (mcp - wrist)
    return out
