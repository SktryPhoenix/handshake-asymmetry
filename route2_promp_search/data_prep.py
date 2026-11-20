"""Step 1: Load Inter-X A001 with raw_betas + per-person ground alignment + G-group split.

Produces a cached dataset of valid handshake reaching sequences:
- Uses raw_betas (real body shapes) instead of zeroed betas.
- Aligns each person independently to z=0 (Z-up) before computing relative geometry.
- Identifies the active hand pair (L/R) and trims the approach-to-contact window.
- Extracts 4-DoF arm angles (shoulder RPY in torso frame + elbow angle) for ProMP,
  and 15-joint upper-body coordinates (pelvis-centered) for LSTM partner prediction.
- Splits strictly by G-group (leave-groups-out) to prevent subject-pair leakage.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import smplx
import torch

from config import (
    CONTACT_DIST_M,
    FEET_JOINTS,
    HEAD,
    INTERX_A001_DIR,
    MCP_L,
    MCP_R,
    MIN_CONTACT_RATIO,
    NECK,
    PELVIS,
    SHOULDER_L,
    SHOULDER_R,
    ELBOW_L,
    ELBOW_R,
    SMPLX_MODEL_PATH,
    SPINE2,
    UPPER_BODY_14,
    WRIST_L,
    WRIST_R,
    ANKLE_L,
    ANKLE_R,
)

_SMPLX_CACHE: dict[str, smplx.SMPLX] = {}


def _get_model(model_path: Path) -> smplx.SMPLX:
    key = str(model_path)
    if key not in _SMPLX_CACHE:
        _SMPLX_CACHE[key] = smplx.create(
            key, model_type="smplx", gender="neutral",
            num_betas=100, num_expression_coeffs=10, use_pca=False,
        )
    return _SMPLX_CACHE[key]


def run_fk_raw(poses: np.ndarray, trans: np.ndarray, raw_betas: np.ndarray,
               model_path: Path, chunk: int = 512) -> np.ndarray:
    """Run SMPL-X FK returning (T, 55, 3) in raw Y-up coordinates."""
    model = _get_model(model_path)
    b = np.zeros(100, dtype=np.float32)
    rb = np.asarray(raw_betas, dtype=np.float32).reshape(-1)
    b[:min(len(rb), 100)] = rb[:100]
    t_len = len(poses)
    out_chunks = []
    with torch.no_grad():
        for s in range(0, t_len, chunk):
            p = torch.from_numpy(np.asarray(poses[s:s + chunk], np.float32))
            tr = torch.from_numpy(np.asarray(trans[s:s + chunk], np.float32))
            n = len(p)
            res = model(
                global_orient=p[:, 0:3].contiguous(),
                body_pose=p[:, 3:66].contiguous(),
                left_hand_pose=p[:, 66:111].contiguous(),
                right_hand_pose=p[:, 111:156].contiguous(),
                jaw_pose=torch.zeros(n, 3),
                leye_pose=torch.zeros(n, 3),
                reye_pose=torch.zeros(n, 3),
                expression=torch.zeros(n, 10),
                betas=torch.from_numpy(b).unsqueeze(0).repeat(n, 1),
                transl=tr,
                return_joints=True,
            )
            out_chunks.append(res.joints[:, :55].cpu().numpy())
    return np.concatenate(out_chunks, axis=0).astype(np.float32)


def to_z_up_and_ground(j_yup: np.ndarray) -> np.ndarray:
    """Convert Y-up (T, J, 3) to Z-up and ground this person's lowest foot at z=0."""
    x, y, z = j_yup[..., 0], j_yup[..., 1], j_yup[..., 2]
    j_zup = np.stack([x, -z, y], axis=-1).astype(np.float32)
    min_z = float(j_zup[:, FEET_JOINTS, 2].min())
    j_zup[..., 2] -= min_z
    return j_zup


def rest_height_m(raw_betas: np.ndarray, model_path: Path) -> float:
    """Rest skeleton height (head - lowest ankle) in meters."""
    model = _get_model(model_path)
    b = np.zeros(100, dtype=np.float32)
    rb = np.asarray(raw_betas, dtype=np.float32).reshape(-1)
    b[:min(len(rb), 100)] = rb[:100]
    with torch.no_grad():
        j = model(betas=torch.from_numpy(b).unsqueeze(0)).joints[0].cpu().numpy()
    return float(j[HEAD, 1] - min(j[ANKLE_L, 1], j[ANKLE_R, 1]))


def palm_positions(j_zup: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "L": 0.5 * (j_zup[:, WRIST_L] + j_zup[:, MCP_L]),
        "R": 0.5 * (j_zup[:, WRIST_R] + j_zup[:, MCP_R]),
    }


def extract_arm_4dof(j_zup: np.ndarray, side: str = "R") -> np.ndarray:
    """Extract 4-DoF arm angles (shoulder pitch, roll, yaw in torso frame + elbow angle).

    Matches Hand Reaching (2103.00616) Sec. III-C: uses shoulder 3-DoF + elbow flexion
    derived from 3D skeleton geometry, leaving wrist angles for IK refinement.
    """
    sh_idx = SHOULDER_R if side == "R" else SHOULDER_L
    el_idx = ELBOW_R if side == "R" else ELBOW_L
    wr_idx = WRIST_R if side == "R" else WRIST_L

    sh = j_zup[:, sh_idx]
    el = j_zup[:, el_idx]
    wr = j_zup[:, wr_idx]

    # Torso local frame at each frame: z_up along spine2->neck, x_lat along L->R shoulder
    z_ax = j_zup[:, NECK] - j_zup[:, SPINE2]
    z_ax /= np.linalg.norm(z_ax, axis=1, keepdims=True) + 1e-8
    lr = j_zup[:, SHOULDER_R] - j_zup[:, SHOULDER_L]
    x_fwd = np.cross(lr, z_ax)
    x_fwd /= np.linalg.norm(x_fwd, axis=1, keepdims=True) + 1e-8
    y_lat = np.cross(z_ax, x_fwd)
    y_lat /= np.linalg.norm(y_lat, axis=1, keepdims=True) + 1e-8

    u = el - sh
    u_len = np.linalg.norm(u, axis=1, keepdims=True) + 1e-8
    u_dir = u / u_len

    # Express upper arm direction in torso frame (fwd, lat, up)
    ux = np.sum(u_dir * x_fwd, axis=1)
    uy = np.sum(u_dir * y_lat, axis=1)
    uz = np.sum(u_dir * z_ax, axis=1)

    sh_pitch = np.arctan2(ux, -uz)                       # forward elevation
    sh_roll = np.arcsin(np.clip(uy, -0.999, 0.999))      # abduction/adduction

    f = wr - el
    f_dir = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    # Elbow flexion angle in [0, pi] (0 = fully extended)
    cos_el = np.clip(np.sum(u_dir * f_dir, axis=1), -0.999, 0.999)
    el_flex = np.arccos(cos_el)

    # Shoulder yaw: angle of forearm projection orthogonal to upper arm
    f_perp = f_dir - np.sum(f_dir * u_dir, axis=1, keepdims=True) * u_dir
    f_perp_norm = np.linalg.norm(f_perp, axis=1, keepdims=True) + 1e-8
    f_perp = f_perp / f_perp_norm
    sh_yaw = np.arctan2(np.sum(f_perp * y_lat, axis=1), np.sum(f_perp * x_fwd, axis=1))

    return np.stack([sh_pitch, sh_roll, sh_yaw, el_flex], axis=1).astype(np.float32)


def find_reaching_window(d_palm: np.ndarray, p_robot_palm: np.ndarray) -> tuple[int, int]:
    """Identify [t_start, t_contact] reaching window up to first sustained contact."""
    touch = np.where(d_palm < CONTACT_DIST_M)[0]
    t_contact = int(touch[0]) if len(touch) > 0 else int(np.argmin(d_palm))
    # Clamp minimum window length so LSTM/ProMP has enough frames
    t_contact = max(t_contact, min(len(d_palm) - 1, 20))
    # Find onset where hand starts moving significantly toward partner (speed > 0.15 m/s)
    vel = np.linalg.norm(np.diff(p_robot_palm[:t_contact + 1], axis=0), axis=1) * 30.0
    moving = np.where(vel > 0.15)[0]
    t_start = int(moving[0]) if len(moving) > 0 else 0
    if t_contact - t_start < 15:
        t_start = max(0, t_contact - 25)
    return t_start, t_contact


def build_cache(data_dir: Path, model_path: Path, out_npz: Path,
                max_pairs: int | None = None) -> dict:
    """Process Inter-X A001 pairs into a clean cached dict saved to out_npz."""
    p1_files = sorted(glob.glob(str(data_dir / "*_P1.npz")))
    if max_pairs is not None:
        p1_files = p1_files[:max_pairs]

    samples = []
    skipped = 0
    for idx, p1_path in enumerate(p1_files):
        p2_path = p1_path[:-7] + "_P2.npz"
        if not os.path.exists(p2_path):
            continue
        seq_name = Path(p1_path).name[:-7]
        m = re.match(r"(G\d+)", seq_name)
        g_id = m.group(1) if m else "G000"

        d1 = np.load(p1_path, allow_pickle=True)
        d2 = np.load(p2_path, allow_pickle=True)
        rb1 = np.asarray(d1["raw_betas"], np.float32)
        rb2 = np.asarray(d2["raw_betas"], np.float32)

        j1 = to_z_up_and_ground(run_fk_raw(d1["poses"], d1["trans"], rb1, model_path))
        j2 = to_z_up_and_ground(run_fk_raw(d2["poses"], d2["trans"], rb2, model_path))
        n = min(len(j1), len(j2))
        j1, j2 = j1[:n], j2[:n]

        p1_palms, p2_palms = palm_positions(j1), palm_positions(j2)
        dists = {f"{a}-{b}": np.linalg.norm(p1_palms[a] - p2_palms[b], axis=1)
                 for a in "LR" for b in "LR"}
        best_pair = min(dists, key=lambda k: float(dists[k].min()))
        d_best = dists[best_pair]
        contact_ratio = float((d_best < CONTACT_DIST_M).mean())
        if contact_ratio < MIN_CONTACT_RATIO:
            skipped += 1
            continue

        s1, s2 = best_pair[0], best_pair[2]
        # Treat P1 as robot-proxy (actor) and P2 as human partner; also keep partner upper body
        t0, tc = find_reaching_window(d_best, p1_palms[s1])
        h1 = rest_height_m(rb1, model_path)
        h2 = rest_height_m(rb2, model_path)

        # Partner upper body (14 symmetric joints) & palm trajectory in world Z-up (grounded at z=0)
        partner_ub = j2[t0:tc + 1, UPPER_BODY_14].astype(np.float32)
        partner_palm = p2_palms[s2][t0:tc + 1].astype(np.float32)
        partner_pelvis = j2[t0:tc + 1, PELVIS].astype(np.float32)
        robot_palm = p1_palms[s1][t0:tc + 1].astype(np.float32)
        robot_pelvis = j1[t0:tc + 1, PELVIS].astype(np.float32)
        arm_4dof = extract_arm_4dof(j1[t0:tc + 1], side=s1).astype(np.float32)

        # Mid-contact clasp point
        touch_mask = d_best < CONTACT_DIST_M
        clasp_pt = 0.5 * (p1_palms[s1][touch_mask] + p2_palms[s2][touch_mask]).mean(axis=0)

        samples.append({
            "seq": seq_name,
            "g_id": g_id,
            "hand_pair": best_pair,
            "h1_m": float(h1),
            "h2_m": float(h2),
            "t_start": int(t0),
            "t_contact": int(tc),
            "partner_ub": partner_ub,         # (L, 15, 3)
            "partner_palm": partner_palm,     # (L, 3)
            "partner_pelvis": partner_pelvis, # (L, 3)
            "robot_palm": robot_palm,         # (L, 3)
            "robot_pelvis": robot_pelvis,     # (L, 3)
            "arm_4dof": arm_4dof,             # (L, 4)
            "clasp_pt": clasp_pt.astype(np.float32),  # (3,)
            "j1_full": j1[t0:tc + 1, :22].astype(np.float32),
            "j2_full": j2[t0:tc + 1, :22].astype(np.float32),
        })
        if (idx + 1) % 50 == 0:
            print(f"[prep] processed {idx + 1}/{len(p1_files)}, kept={len(samples)}", flush=True)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npz, np.array(samples, dtype=object), allow_pickle=True)
    print(f"[prep] done: kept={len(samples)} valid pairs, skipped={skipped} (<5% contact) -> {out_npz}")
    return {"kept": len(samples), "skipped": skipped, "path": str(out_npz)}


def split_by_group(samples: list[dict], seed: int = 42,
                   val_frac: float = 0.15, test_frac: float = 0.20) -> tuple[list[dict], list[dict], list[dict]]:
    """Leak-free Group-level split (each G-group is one unique human pair)."""
    groups = sorted({s["g_id"] for s in samples})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    n_test = max(1, int(round(len(groups) * test_frac)))
    n_val = max(1, int(round(len(groups) * val_frac)))
    test_g = {groups[i] for i in perm[:n_test]}
    val_g = {groups[i] for i in perm[n_test:n_test + n_val]}

    train = [s for s in samples if s["g_id"] not in test_g and s["g_id"] not in val_g]
    val = [s for s in samples if s["g_id"] in val_g]
    test = [s for s in samples if s["g_id"] in test_g]
    return train, val, test


def load_cache(cache_path: Path) -> list[dict]:
    return list(np.load(cache_path, allow_pickle=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=INTERX_A001_DIR)
    ap.add_argument("--model", type=Path, default=SMPLX_MODEL_PATH)
    ap.add_argument("--out", type=Path, default=Path("/mnt/sdc/ltr/runs/handshake_route2/a001_cache.npy"))
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()
    build_cache(args.data_dir, args.model, args.out, max_pairs=args.max_pairs)
