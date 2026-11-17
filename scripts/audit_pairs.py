#!/usr/bin/env python3
"""Twelve-point quality audit for *paired* (two-person) motion data.

Why audit before modelling
--------------------------
It is common for released interaction datasets to look fine and be quietly unusable:

  * every subject fitted to the SAME body (the ``betas`` field is all zeros), so the
    "body-size mismatch" one wants to study simply does not exist in the data;
  * the pair is grounded jointly, leaving one partner floating several centimetres;
  * "contact" is rare, so a hard contact constraint is conceptually wrong.

This script measures those physical quantities instead of trusting metadata.

Metrics
-------
 1  frames / fps
 2  per-caregiver limb ratio under zero betas     (expect exactly 1.000 -> same body)
 3  limb ratio under ``raw_betas``                (the real body-size difference)
 4  rest height from ``raw_betas`` (left/right)   (the destroyed height variation)
 5  palm-to-palm minimum distance                 (is there contact at all)
 6  which hand pair is used (L-L / L-R / R-L / R-R)
 7  contact-frame ratio                           (contact is an event, not a state)
 8  palm height at contact                        (handshake ~90 cm vs hug/dance)
 9  lowest joint height                           (global floor offset)
10  ``conversion_y_shift`` consistency            (can metadata be trusted)
11  palm acceleration jitter
12  foot slide per frame

Usage
-----
    python audit_pairs.py --root /path/to/a001 --model SMPLX_NEUTRAL.npz \
        --out audit.json --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from smplx_fk import FEET, JOINT, load_motion, palms, run_fk

_HEIGHT_AXIS = 1        # FK output is Y-up: height is component 1, NOT 2
CONTACT_CM = 10.0


def arm_length(betas, model_path: str) -> float:
    import smplx
    import torch
    m = smplx.create(model_path, model_type="smplx", num_betas=16, use_pca=False, batch_size=1)
    with torch.no_grad():
        j = m(betas=torch.tensor(np.asarray(betas, np.float32).reshape(1, -1))).joints[0].numpy()
    return float(np.linalg.norm(j[18] - j[16]) + np.linalg.norm(j[20] - j[18]))


def rest_height(betas, model_path: str) -> float:
    import smplx
    import torch
    m = smplx.create(model_path, model_type="smplx", num_betas=16, use_pca=False, batch_size=1)
    with torch.no_grad():
        j = m(betas=torch.tensor(np.asarray(betas, np.float32).reshape(1, -1))).joints[0].numpy()
    return float(j[15, 1] - min(j[7, 1], j[8, 1]))


def _bones(j: np.ndarray) -> dict:
    def m(a: int, b: int) -> float:
        return float(np.linalg.norm(j[:, a] - j[:, b], axis=1).mean())
    return {"upper_l": m(JOINT["left_shoulder"], JOINT["left_elbow"]),
            "fore_l": m(JOINT["left_elbow"], JOINT["left_wrist"])}


def audit_one(job) -> dict | None:
    f1, f2, model_path = job
    try:
        d1, d2 = load_motion(f1), load_motion(f2)
    except Exception:
        return None
    n = min(d1["n_frames"], d2["n_frames"])
    zero = np.zeros(16, np.float32)
    J1 = run_fk(d1["poses"][:n], d1["trans"][:n], zero, model_path)
    J2 = run_fk(d2["poses"][:n], d2["trans"][:n], zero, model_path)

    p1, p2 = palms(J1), palms(J2)
    dists = {"%s-%s" % (a, b): np.linalg.norm(p1[a] - p2[b], axis=1) for a in "LR" for b in "LR"}
    best = min(dists, key=lambda k: float(dists[k].min()))
    dmin = float(dists[best].min())
    touch = dists[best] < CONTACT_CM / 100.0

    row = {
        "pair": Path(f1).name.replace("_P1.npz", ""),
        "n_frames": int(n),
        "fps": d1["fps"],
        "has_raw_betas": d1["raw_betas"] is not None and d2["raw_betas"] is not None,
        "palm_min_cm": dmin * 100,
        "palm_pair": best,
        "contact_ratio": float(touch.mean()),
        "palm_height_cm": float(np.median(np.r_[p1[best[0]][:, _HEIGHT_AXIS],
                                               p2[best[2]][:, _HEIGHT_AXIS]])) * 100,
        "floor_min_cm": float(min(J1[:, :22, _HEIGHT_AXIS].min(),
                                  J2[:, :22, _HEIGHT_AXIS].min())) * 100,
        "y_shift_cm": -d1.get("y_shift", float("nan")) * 100,
    }
    if touch.any():
        row["palm_height_touch_cm"] = float(np.median(np.r_[
            p1[best[0]][touch][:, _HEIGHT_AXIS], p2[best[2]][touch][:, _HEIGHT_AXIS]])) * 100

    v = np.diff(p1[best[0]], axis=0)
    row["jitter_cm"] = float(np.median(np.linalg.norm(np.diff(v, axis=0), axis=1))) * 100 if len(v) > 1 else 0.0

    slips = []
    for J in (J1, J2):
        for jid in (JOINT["left_ankle"], JOINT["right_ankle"], JOINT["left_foot"], JOINT["right_foot"]):
            y = J[:, jid, _HEIGHT_AXIS]
            low = y < y.min() + 0.03
            xy = J[low][:, jid][:, [0, 2]]
            if len(xy) > 1:
                slips.append(float(np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    row["foot_slide_cm_per_frame"] = float(np.median(slips)) * 100 if slips else float("nan")

    b1, b2 = _bones(J1), _bones(J2)
    for k in ("upper_l", "fore_l"):
        row["ratio_zero_" + k] = b2[k] / b1[k]
    if row["has_raw_betas"]:
        J1r = run_fk(d1["poses"][:n], d1["trans"][:n], d1["raw_betas"], model_path)
        J2r = run_fk(d2["poses"][:n], d2["trans"][:n], d2["raw_betas"], model_path)
        br1, br2 = _bones(J1r), _bones(J2r)
        for k in ("upper_l", "fore_l"):
            row["ratio_raw_" + k] = br2[k] / br1[k]
        row["h1_raw_cm"] = rest_height(d1["raw_betas"], model_path) * 100
        row["h2_raw_cm"] = rest_height(d2["raw_betas"], model_path) * 100
        row["h_ratio_raw"] = row["h2_raw_cm"] / row["h1_raw_cm"]
    return row


def collect(root: Path) -> list:
    """Only pair files that live in a shared (pair) directory.

    Datasets sometimes ship both ``*_1p`` and ``*_2p`` variants. The ``1p`` files are
    each fitted in their OWN coordinate frame, so pairing them is meaningless
    (a handshake would show an 8 cm pelvis separation). Only ``*_2p`` is usable.
    """
    jobs = []
    for p1 in sorted(root.rglob("P1.npz")):
        rel = str(p1)
        if "_1p" in rel and "_2p" not in rel:
            continue
        p2 = p1.with_name("P2.npz")
        if p2.exists():
            jobs.append((str(p1), str(p2)))
    for p1 in sorted(root.glob("*_P1.npz")):
        p2 = Path(str(p1).replace("_P1.npz", "_P2.npz"))
        if p2.exists():
            jobs.append((str(p1), str(p2)))
    return jobs


def summarize(rows: list[dict]) -> dict:
    def col(k):
        return np.array([r[k] for r in rows if k in r and r[k] == r[k]], float)
    out = {"n_pairs": len(rows), "n_frames_total": int(sum(r["n_frames"] for r in rows))}
    for k in ("palm_min_cm", "palm_height_cm", "palm_height_touch_cm", "floor_min_cm",
              "jitter_cm", "foot_slide_cm_per_frame", "y_shift_cm"):
        a = col(k)
        if a.size:
            out[k] = dict(med=float(np.median(a)), p10=float(np.percentile(a, 10)),
                          p90=float(np.percentile(a, 90)))
    for k in ("ratio_zero_upper_l", "ratio_zero_fore_l", "ratio_raw_upper_l",
              "ratio_raw_fore_l", "h_ratio_raw"):
        a = col(k)
        if a.size:
            out[k] = dict(med=float(np.median(a)), mean_abs_dev_pct=float(np.mean(np.abs(a - 1)) * 100),
                          min=float(a.min()), max=float(a.max()))
    out["contact_ratio"] = float(np.mean([r["contact_ratio"] for r in rows]))
    out["pairs_with_contact_ge5pct"] = int(sum(1 for r in rows if r["contact_ratio"] >= 0.05))
    cnt: dict = {}
    for r in rows:
        cnt[r["palm_pair"]] = cnt.get(r["palm_pair"], 0) + 1
    out["palm_pair_dist"] = cnt
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("audit.json"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = collect(args.root)
    print("[audit] %s -> %d pairs" % (args.root, len(jobs)))
    with mp.Pool(args.workers) as pool:
        rows = [r for r in pool.imap_unordered(audit_one, [(a, b, args.model) for a, b in jobs],
                                               chunksize=2) if r]
    print("[audit] %d pairs audited" % len(rows))
    res = {"root": str(args.root), "summary": summarize(rows), "rows": rows}
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()
