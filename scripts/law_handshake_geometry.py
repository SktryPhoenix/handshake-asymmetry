#!/usr/bin/env python3
"""Measure the *negotiation law* of real human-human handshakes.

The question this answers
------------------------
When two people of different height shake hands, **how is the mismatch split between
them?** Specifically:

  - where does the clasp sit vertically (absolute height vs. scaled height)?
  - where does it sit horizontally along the line joining the two bodies?
  - how far apart do they stand?
  - how much torso lean and arm extension does each contribute?

Output is a table bucketed by height difference, plus a JSON dump.

Inputs
------
A directory of paired `.npz` files named `*_P1.npz` / `*_P2.npz`, each holding
`poses (T,156)`, `trans (T,3)`, `betas (16,)`, and ideally `raw_betas (16,)`
(Inter-X / MotionHub mirror layout). `raw_betas` is REQUIRED for correct body size:
the released `betas` field is all zeros (every subject is the same neutral body).

Usage
-----
    python law_handshake_geometry.py \
        --data-root /path/to/a001 \
        --model /path/to/SMPLX_NEUTRAL.npz \
        --out law.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smplx_fk import FEET, JOINT, load_motion, palms, run_fk


def arm_length(betas, model_path: str) -> float:
    """Upper-arm + forearm length in the rest pose (shape only, pose-independent)."""
    import smplx
    import torch
    model = smplx.create(model_path, model_type="smplx", num_betas=16,
                         use_pca=False, batch_size=1)
    with torch.no_grad():
        j = model(betas=torch.tensor(np.asarray(betas, np.float32)).reshape(1, -1)).joints[0].numpy()
    return float(np.linalg.norm(j[18] - j[16]) + np.linalg.norm(j[20] - j[18]))


def lean_deg(j, shoulder: int, frames) -> float:
    """Angle between (shoulder - pelvis) and the vertical axis, in degrees."""
    v = j[frames, shoulder] - j[frames, JOINT["pelvis"]]
    cos = v[:, 1] / (np.linalg.norm(v, axis=1) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, 0, 1))).mean())


def one_pair(f1: Path, f2: Path, model_path: str, contact_cm: float = 8.0):
    d1, d2 = load_motion(f1), load_motion(f2)
    if d1["raw_betas"] is None or d2["raw_betas"] is None:
        return None                      # body size unrecoverable -> skip
    n = min(d1["n_frames"], d2["n_frames"])
    J1 = run_fk(d1["poses"][:n], d1["trans"][:n], d1["raw_betas"], model_path)
    J2 = run_fk(d2["poses"][:n], d2["trans"][:n], d2["raw_betas"], model_path)
    # Each person is grounded by their OWN lowest foot. The released data only aligns
    # the pair jointly, which leaves one partner floating by ~6 cm.
    for J in (J1, J2):
        J[:, :, 1] -= J[:, FEET, 1].min()

    p1, p2 = palms(J1), palms(J2)
    pair = min(((a, b) for a in "LR" for b in "LR"),
               key=lambda ab: float(np.linalg.norm(p1[ab[0]] - p2[ab[1]], axis=1).min()))
    dist = np.linalg.norm(p1[pair[0]] - p2[pair[1]], axis=1)
    touch = dist < contact_cm / 100.0
    if touch.sum() < 3:
        return None

    A1 = arm_length(d1["raw_betas"], model_path)
    A2 = arm_length(d2["raw_betas"], model_path)
    h1 = float(np.percentile(J1[:, JOINT["head"], 1], 95))
    h2 = float(np.percentile(J2[:, JOINT["head"], 1], 95))

    clasp = 0.5 * (p1[pair[0]][touch] + p2[pair[1]][touch])
    z_c = float(np.median(clasp[:, 1]))

    # horizontal quantities live in components (0, 2) because FK output is Y-up
    pel1 = J1[touch][:, JOINT["pelvis"]][:, [0, 2]]
    pel2 = J2[touch][:, JOINT["pelvis"]][:, [0, 2]]
    standoff = float(np.median(np.linalg.norm(pel1 - pel2, axis=1)))
    seg = pel2 - pel1
    lam = np.einsum("ij,ij->i", clasp[:, [0, 2]] - pel1, seg) / (np.linalg.norm(seg, axis=1) ** 2 + 1e-9)
    lam = float(np.median(lam))

    sho = {1: JOINT["left_shoulder" if pair[0] == "L" else "right_shoulder"],
           2: JOINT["left_shoulder" if pair[1] == "L" else "right_shoulder"]}
    wr = {1: JOINT["left_wrist" if pair[0] == "L" else "right_wrist"],
          2: JOINT["left_wrist" if pair[1] == "L" else "right_wrist"]}
    z_s1 = float(np.median(J1[touch][:, sho[1], 1]))
    z_s2 = float(np.median(J2[touch][:, sho[2], 1]))
    e1 = float(np.mean(np.linalg.norm(J1[touch][:, wr[1]] - J1[touch][:, sho[1]], axis=1)) / A1)
    e2 = float(np.mean(np.linalg.norm(J2[touch][:, wr[2]] - J2[touch][:, sho[2]], axis=1)) / A2)
    return dict(clip=f1.stem.replace("_P1", ""), n=int(n), h1=h1, h2=h2, z_clasp=z_c,
                z_shoulder_1=z_s1, z_shoulder_2=z_s2, standoff=standoff, lam=lam,
                extension_1=e1, extension_2=e2,
                lean_1=lean_deg(J1, sho[1], touch), lean_2=lean_deg(J2, sho[2], touch),
                palm_pair="%s-%s" % pair, contact_ratio=float(touch.mean()))


BUCKETS = [(-1.0, -0.10), (-0.10, -0.04), (-0.04, 0.04), (0.04, 0.10), (0.10, 1.0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--model", required=True, help="SMPLX_NEUTRAL.npz")
    ap.add_argument("--out", type=Path, default=Path("law.json"))
    ap.add_argument("--contact-cm", type=float, default=8.0)
    args = ap.parse_args()

    pairs = []
    for p1 in sorted(args.data_root.glob("*_P1.npz")):
        p2 = Path(str(p1).replace("_P1.npz", "_P2.npz"))
        if p2.exists():
            pairs.append((p1, p2))
    print("[law] %d candidate pairs" % len(pairs))

    rows = []
    for f1, f2 in pairs:
        r = one_pair(f1, f2, args.model, args.contact_cm)
        if r:
            rows.append(r)
    print("[law] %d pairs with real contact (<%.0f cm)" % (len(rows), args.contact_cm))
    if not rows:
        return

    a = {k: np.array([r[k] for r in rows]) for k in rows[0] if isinstance(rows[0][k], (int, float))}
    print("\n%-14s %5s %7s %9s %9s %9s %9s" % (
        "dh (m)", "n", "h2/h1", "clasp_z", "clasp/矮肩", "矮伸%", "矮倾°"))
    for lo, hi in BUCKETS:
        dh = a["h2"] - a["h1"]
        m = (dh >= lo) & (dh < hi)
        if m.sum() == 0:
            continue
        short = np.where(a["h1"][m] < a["h2"][m], 1, 2)
        zs = np.where(short == 1, a["z_shoulder_1"][m], a["z_shoulder_2"][m])
        e_short = np.where(short == 1, a["extension_1"][m], a["extension_2"][m])
        l_short = np.where(short == 1, a["lean_1"][m], a["lean_2"][m])
        print("%-14s %5d %7.3f %8.1f %9.3f %9.1f %9.1f" % (
            "[%+.2f,%+.2f)" % (lo, hi), m.sum(), np.mean(a["h2"][m] / a["h1"][m]),
            a["z_clasp"][m].mean() * 100, float(np.mean(a["z_clasp"][m] / zs)),
            float(np.mean(e_short)) * 100, float(np.mean(l_short))))

    print("\nclasp position along pelvis-pelvis line (0=p1, 0.5=midpoint): "
          "median %.3f" % float(np.median(a["lam"])))
    print("standoff (pelvis distance): median %.1f cm" % (float(np.median(a["standoff"])) * 100))
    print("extension: median %.1f%%" % (float(np.median(np.r_[a["extension_1"], a["extension_2"]])) * 100))
    print("lean: median %.1f deg" % float(np.median(np.r_[a["lean_1"], a["lean_2"]])))

    args.out.write_text(json.dumps({"n_pairs": len(rows), "rows": rows}, indent=1))
    print("\n-> %s" % args.out)


if __name__ == "__main__":
    main()
