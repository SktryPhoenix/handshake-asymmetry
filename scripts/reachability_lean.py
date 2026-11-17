#!/usr/bin/env python3
"""How much arm reach does the mismatch cost, and how much does torso lean recover?

Motivation
----------
A humanoid that stands in for one of the partners must reach the *other person's* hand.
Because a humanoid's arm is proportionally shorter than a human's, some contact frames
are geometrically unreachable unless the robot leans forward - which is exactly what
humans do when shaking hands (measured lean in this repo: 22-28 degrees).

This script quantifies, over real handshake contact frames:

  1. required shoulder->clasp distance for several robot lean angles
  2. the fraction of contact frames exceeding the robot's arm length
  3. for the residual frames, how much concession is needed
     (either stepping closer, or lowering the clasp)

Robot constants below were measured from `g1_29dof.xml` (Unitree G1) and the
SMPL-X neutral template; they are inputs, not claims about particular hardware.

Usage
-----
    python reachability_lean.py --data-root /path/to/a001 \
        --model SMPLX_NEUTRAL.npz --out reach.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smplx_fk import FEET, JOINT, load_motion, palms, run_fk

# --- constants (see docs/02-handshake-geometry-law.md) ---
G1_ARM_REACH = 0.3873        # shoulder->wrist fully extended: 0.2031 + 0.1843
G1_SHOULDER_H = 1.0848       # shoulder height above ground when standing
HUMAN_SHOULDER_H = 1.314     # SMPL-X neutral shoulder height above ground
SHOULDER_DROP = HUMAN_SHOULDER_H - G1_SHOULDER_H   # 0.2292 m vertical mismatch
LEANS_DEG = (0, 15, 25, 35)
CONCESSION_LEAN_DEG = 25     # lean actually observed in human handshakes


def contact_frames(f1: Path, f2: Path, model_path: str, contact_cm: float = 8.0):
    """Return (clasp positions, actor shoulder, actor pelvis) at contact frames."""
    d1, d2 = load_motion(f1), load_motion(f2)
    if d1["raw_betas"] is None or d2["raw_betas"] is None:
        return None
    n = min(d1["n_frames"], d2["n_frames"])
    J1 = run_fk(d1["poses"][:n], d1["trans"][:n], d1["raw_betas"], model_path)
    J2 = run_fk(d2["poses"][:n], d2["trans"][:n], d2["raw_betas"], model_path)
    for J in (J1, J2):
        J[:, :, 1] -= J[:, FEET, 1].min()

    p1, p2 = palms(J1), palms(J2)
    pair = min(((a, b) for a in "LR" for b in "LR"),
               key=lambda ab: float(np.linalg.norm(p1[ab[0]] - p2[ab[1]], axis=1).min()))
    touch = np.linalg.norm(p1[pair[0]] - p2[pair[1]], axis=1) < contact_cm / 100.0
    if touch.sum() < 3:
        return None
    clasp = 0.5 * (p1[pair[0]][touch] + p2[pair[1]][touch])
    actor_shoulder = J1[touch][:, JOINT["left_shoulder" if pair[0] == "L" else "right_shoulder"]]
    actor_pelvis = J1[touch][:, JOINT["pelvis"]]
    return clasp, actor_shoulder, actor_pelvis


def reach_with_lean(clasp, shoulder, pelvis, lean_deg: float, drop: float = SHOULDER_DROP):
    """Required shoulder->clasp distance after replacing the actor with a shorter robot.

    The shoulder is lowered by `drop` (height mismatch), then rotated forward about the
    pelvis by `lean_deg` towards the clasp. Rotating adds horizontal reach but lowers
    the shoulder slightly; both effects are modelled here.
    """
    sh = shoulder.copy()
    sh[:, 1] -= drop
    L = np.linalg.norm(sh - pelvis, axis=1)
    v = clasp[:, [0, 2]] - pelvis[:, [0, 2]]
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    th = np.radians(lean_deg)
    sh[:, [0, 2]] += (L * np.sin(th))[:, None] * v
    sh[:, 1] -= L * (1 - np.cos(th))
    return np.linalg.norm(clasp - sh, axis=1), sh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("reach.json"))
    ap.add_argument("--contact-cm", type=float, default=8.0)
    args = ap.parse_args()

    pairs = []
    for p1 in sorted(args.data_root.glob("*_P1.npz")):
        p2 = Path(str(p1).replace("_P1.npz", "_P2.npz"))
        if p2.exists():
            pairs.append((p1, p2))
    print("[reach] %d candidate pairs" % len(pairs))

    per_lean = {t: [] for t in LEANS_DEG}
    step_in, drop_req, n_pairs = [], [], 0
    for f1, f2 in pairs:
        got = contact_frames(f1, f2, args.model, args.contact_cm)
        if got is None:
            continue
        n_pairs += 1
        clasp, shoulder, pelvis = got
        for t in LEANS_DEG:
            r, _ = reach_with_lean(clasp, shoulder, pelvis, t)
            per_lean[t].extend(r.tolist())
        _, sh = reach_with_lean(clasp, shoulder, pelvis, CONCESSION_LEAN_DEG)
        d = clasp - sh
        hor = np.linalg.norm(d[:, [0, 2]], axis=1)
        ver = d[:, 1]
        over = hor ** 2 + ver ** 2 > G1_ARM_REACH ** 2
        s_req = np.where(over, hor - np.sqrt(np.maximum(G1_ARM_REACH ** 2 - ver ** 2, 0)), 0)
        z_req = np.where(over, ver - np.sqrt(np.maximum(G1_ARM_REACH ** 2 - hor ** 2, 0)), 0)
        step_in.extend((s_req * 100).tolist())
        drop_req.extend((z_req * 100).tolist())

    print("[reach] %d pairs with contact" % n_pairs)
    print("\n%-8s %9s %9s %12s %11s" % ("lean", "median", "p90", "over_reach", "reachable"))
    table = {}
    for t in LEANS_DEG:
        r = np.array(per_lean[t])
        if r.size == 0:
            continue
        over = float((r > G1_ARM_REACH).mean()) * 100
        table[str(t)] = dict(n=int(r.size), median_cm=float(np.median(r)) * 100,
                             p90_cm=float(np.percentile(r, 90)) * 100,
                             over_pct=over, reachable_pct=100 - over)
        print("%-8s %8.1f %9.1f %11.0f%% %10.0f%%" % (
            "%d deg" % t, float(np.median(r)) * 100, float(np.percentile(r, 90)) * 100,
            over, 100 - over))

    a, b = np.array(step_in), np.array(drop_req)
    m = np.minimum(a, b)
    pos = m[m > 0]
    print("\nafter %d deg lean, residual frames need a concession:" % CONCESSION_LEAN_DEG)
    print("  frames needing one: %.0f%%" % (100 * float((m > 0).mean())))
    print("  step closer  : median %.1f cm" % float(np.median(a[a > 0])))
    print("  lower clasp  : median %.1f cm" % float(np.median(b[b > 0])))
    print("  cheaper of both: median %.1f cm, p90 %.1f cm" % (
        float(np.median(pos)), float(np.percentile(pos, 90))))

    args.out.write_text(json.dumps(dict(
        n_pairs=n_pairs, g1_arm_reach_m=G1_ARM_REACH, shoulder_drop_m=SHOULDER_DROP,
        per_lean=table,
        concession_at_25deg=dict(frames_needing_pct=100 * float((m > 0).mean()),
                                 median_cm=float(np.median(pos)),
                                 p90_cm=float(np.percentile(pos, 90)),
                                 step_in_median_cm=float(np.median(a[a > 0])),
                                 lower_clasp_median_cm=float(np.median(b[b > 0])),
                                 ),
    ), indent=1))
    print("\n-> %s" % args.out)


if __name__ == "__main__":
    main()
