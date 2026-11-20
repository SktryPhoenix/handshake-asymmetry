"""Step 6: End-to-End Pipeline & 4-Way Ablation for Route 2.

Runs:
1. Data preparation (or loads cached a001_cache.npy with raw_betas + per-person grounding).
2. Leak-free G-group split (train / val / test).
3. Layer P: Fit ProMP over 4-DoF arm reaching trajectories (mirrored to unified frame).
4. Layer D: Train causal PartnerHandLSTM to predict partner's final hand position.
5. Layer R+D Evaluation on held-out test G-groups across 4 conditions:
   - A: Naive Baseline (K=1, 0 deg lean, no prediction -> tracks delayed current palm)
   - B: Paper A (Hand Reaching: LSTM + Sigmoid blend + ProMP mean + DLS IK, 0 deg lean)
   - C: Paper A + B (Embodiment-Aware Search K=16 over ProMP + posture pool)
   - D: Route 2 + Negotiation Law (Paper A + B + 25 deg lean + -0.22*dh z-law + 84-90% band)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_WORK_DIR,
    HISTORY_FRAMES,
    INTERX_A001_DIR,
    LAW_EXT_TARGET_HIGH,
    LAW_EXT_TARGET_LOW,
    SMPLX_MODEL_PATH,
)
from data_prep import build_cache, load_cache, split_by_group
from g1_kinematics import G1Kinematics
from predictor_lstm import blend_target, train_lstm_predictor
from promp import ProMP
from reaction_search import search_best_reaction


def _unify_arm_4dof(arm_4dof: np.ndarray, side: str) -> np.ndarray:
    out = arm_4dof.copy()
    if side == "L":
        out[:, 1] *= -1.0  # sh_roll
        out[:, 2] *= -1.0  # sh_yaw
    return out


def _apply_promp_side(arm_4dof: np.ndarray, side: str) -> np.ndarray:
    return _unify_arm_4dof(arm_4dof, side)


def _eval_single_trajectory(
    g1: G1Kinematics,
    arm_4dof_traj: np.ndarray,
    sample: dict,
    target_palms: np.ndarray,
    lean_deg: float,
    standoff_m: float = 0.84,
) -> dict:
    side = sample["hand_pair"][0]
    roll = g1.rollout_trajectory(
        partner_pelvis=sample["partner_pelvis"],
        robot_pelvis=sample["robot_pelvis"],
        target_palms=target_palms,
        arm_4dof_traj=_apply_promp_side(arm_4dof_traj, side),
        side=side,
        standoff_m=standoff_m,
        lean_deg=lean_deg,
    )
    feas = g1.evaluate_feasibility(roll["q_traj"], roll["palm_traj"], roll["sh_traj"])
    final_err_cm = float(np.linalg.norm(roll["palm_traj"][-1] - sample["partner_palm"][-1]) * 100.0)
    sh_to_target = float(np.linalg.norm(sample["partner_palm"][-1] - roll["sh_traj"][-1]))
    concession_cm = max(0.0, sh_to_target - 0.3873) * 100.0
    return {
        "final_err_cm": final_err_cm,
        "concession_cm": concession_cm,
        "feas": feas,
        "q_traj": roll["q_traj"],
        "palm_traj": roll["palm_traj"],
        "lean_deg": float(lean_deg),
        "standoff_m": float(standoff_m),
    }


def _summarize(records: list[dict], label: str) -> dict:
    errs = np.array([r["final_err_cm"] for r in records])
    concs = np.array([r["concession_cm"] for r in records])
    valids = np.array([r["feas"]["r_valid"] for r in records], dtype=float)
    exts = np.array([r["feas"]["ext_ratio_final"] for r in records])
    jerks = np.array([r["feas"]["jerk_cm"] for r in records])
    # Contact success: final palm error <= 5.0 cm AND robot feasible
    succ = np.array([(r["final_err_cm"] <= 5.0 and r["feas"]["r_valid"]) for r in records], dtype=float)
    in_band = np.array(
        [(LAW_EXT_TARGET_LOW <= r["feas"]["ext_ratio_final"] <= LAW_EXT_TARGET_HIGH) for r in records],
        dtype=float,
    )
    return {
        "condition": label,
        "n_test": len(records),
        "palm_err_med_cm": round(float(np.median(errs)), 2),
        "palm_err_mean_cm": round(float(np.mean(errs)), 2),
        "success_le5cm_pct": round(float(succ.mean() * 100.0), 1),
        "r_valid_pct": round(float(valids.mean() * 100.0), 1),
        "ext_ratio_med": round(float(np.median(exts)), 3),
        "ext_in_comfort_pct": round(float(in_band.mean() * 100.0), 1),
        "jerk_med_cm": round(float(np.median(jerks)), 3),
        "concession_med_cm": round(float(np.median(concs)), 2),
        "concession_p90_cm": round(float(np.percentile(concs, 90)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--max-pairs", type=int, default=None, help="Optional subset for quick smoke tests")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--k-search", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or (args.work_dir / "a001_cache.npy")

    if not cache_path.exists():
        print("[1/5] Building A001 cache (raw_betas + per-person grounding)...")
        build_cache(INTERX_A001_DIR, SMPLX_MODEL_PATH, cache_path, max_pairs=args.max_pairs)
    else:
        print(f"[1/5] Loading existing cache: {cache_path}")

    samples = load_cache(cache_path)
    train_s, val_s, test_s = split_by_group(samples, seed=args.seed)
    print(f"      Split (by G-group): train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    # [2/5] Fit ProMP prior on train split
    print("[2/5] Fitting 4-DoF ProMP prior...")
    train_arm_trajs = [_unify_arm_4dof(s["arm_4dof"], s["hand_pair"][0]) for s in train_s]
    promp = ProMP(n_dims=4, n_basis=8, width=0.14, reg_lambda=1e-6).fit(train_arm_trajs)
    promp.save(str(args.work_dir / "promp_a001.npz"))

    # [3/5] Train causal PartnerHandLSTM
    print("[3/5] Training causal PartnerHandLSTM...")
    predictor, lstm_metrics = train_lstm_predictor(
        train_s, val_s, history=HISTORY_FRAMES, epochs=args.epochs, seed=args.seed,
    )
    print("      LSTM val metrics:", lstm_metrics)

    # Evaluate causal prediction error on held-out test split
    test_pred_errs, test_nopred_errs = [], []
    for s in test_s:
        h_hat = predictor.predict_sequence(s, history=HISTORY_FRAMES)
        h_final = s["partner_palm"][-1]
        test_pred_errs.extend(np.linalg.norm(h_hat - h_final[None, :], axis=1) * 100.0)
        test_nopred_errs.extend(np.linalg.norm(s["partner_palm"] - h_final[None, :], axis=1) * 100.0)
    print(f"      Test causal hand prediction error: LSTM={np.mean(test_pred_errs):.2f} cm "
          f"vs NoPred={np.mean(test_nopred_errs):.2f} cm")

    # [4/5] Run 4-way evaluation on test split
    print(f"[4/5] Running 4-way evaluation on {len(test_s)} held-out test sequences...")
    g1 = G1Kinematics()
    rng = np.random.default_rng(args.seed)

    res_a, res_b, res_c, res_d = [], [], [], []
    vis_records = []
    for i, s in enumerate(test_s):
        t_len = len(s["partner_palm"])
        side = s["hand_pair"][0]
        h_curr = s["partner_palm"]
        h_hat = predictor.predict_sequence(s, history=HISTORY_FRAMES)

        # Blended online target (Paper A Eq. 6)
        h_blend = np.stack([
            blend_target(h_hat[t], h_curr[t], t, t_est=t_len) for t in range(t_len)
        ], axis=0)

        # Simulated 4-frame (~133ms) perception delay for Naive Baseline A
        delay = 4
        idx_del = np.clip(np.arange(t_len) - delay, 0, t_len - 1)
        h_delayed = h_curr[idx_del]

        promp_mean = promp.rollout_mean(t_len)

        # A: Naive baseline (delayed target, 0 deg lean)
        ra = _eval_single_trajectory(g1, promp_mean, s, h_delayed, lean_deg=0.0)
        res_a.append(ra)

        # B: Paper A (LSTM + Sigmoid blend + ProMP conditioned at z=1.0, 0 deg lean)
        # Solve anatomical 4-DoF goal for h_blend[-1] and condition ProMP prior
        dir_xy = s["robot_pelvis"][-1, :2] - s["partner_pelvis"][-1, :2]
        n_xy = float(np.linalg.norm(dir_xy))
        dir_xy = dir_xy / n_xy if n_xy > 1e-4 else np.array([1.0, 0.0])
        r_xy_b = s["partner_pelvis"][-1, :2] + dir_xy * 0.84
        yaw_b = float(np.arctan2(-dir_xy[1], -dir_xy[0]))
        g1_goal_arm4 = g1.solve_goal_arm4(r_xy_b, yaw_b, h_blend[-1], promp_mean[-1], side=side, lean_deg=0.0)
        hum_goal_arm4 = _unify_arm_4dof(g1.g1_to_human_4dof(g1_goal_arm4)[None, :], side)[0]
        mu_cond, _ = promp.condition(1.0, hum_goal_arm4, obs_std=0.08)
        promp_cond = promp.decode(mu_cond, t_len)
        rb = _eval_single_trajectory(g1, promp_cond, s, h_blend, lean_deg=0.0)
        res_b.append(rb)

        # C: Paper A + Paper B (K=16 Reaction Search without negotiation law)
        best_c = search_best_reaction(
            g1=g1, promp=promp, partner_pelvis=s["partner_pelvis"],
            robot_pelvis=s["robot_pelvis"],
            partner_palm_true=h_curr, target_palms=h_blend,
            side=side, k_candidates=args.k_search, use_negotiation_prior=False, rng=rng,
        )
        res_c.append(best_c)

        # D: Route 2 + Negotiation Law (25 deg lean + standoff + 84-90% comfort band)
        # Keep the same partner target h_blend for an apples-to-apples reach comparison
        best_d = search_best_reaction(
            g1=g1, promp=promp, partner_pelvis=s["partner_pelvis"],
            robot_pelvis=s["robot_pelvis"],
            partner_palm_true=h_curr, target_palms=h_blend,
            side=side, k_candidates=args.k_search, use_negotiation_prior=True, rng=rng,
        )
        res_d.append(best_d)

        vis_records.append({
            "seq": s["seq"],
            "hand_pair": s["hand_pair"],
            "h1_m": s["h1_m"],
            "h2_m": s["h2_m"],
            "j1_full": s["j1_full"],
            "j2_full": s["j2_full"],
            "partner_palm": h_curr,
            "lstm_pred": h_hat.astype(np.float32),
            "blend_target": h_blend.astype(np.float32),
            "schemes": {
                "A_朴素延迟_0度": {
                    "q_traj": ra["q_traj"].astype(np.float32),
                    "palm_traj": ra["palm_traj"].astype(np.float32),
                    "err_cm": ra["final_err_cm"],
                    "valid": ra["feas"]["r_valid"],
                    "ext": ra["feas"]["ext_ratio_final"],
                    "lean_deg": ra["lean_deg"],
                    "standoff_m": ra["standoff_m"],
                },
                "B_PaperA_LSTM+ProMP_0度": {
                    "q_traj": rb["q_traj"].astype(np.float32),
                    "palm_traj": rb["palm_traj"].astype(np.float32),
                    "err_cm": rb["final_err_cm"],
                    "valid": rb["feas"]["r_valid"],
                    "ext": rb["feas"]["ext_ratio_final"],
                    "lean_deg": rb["lean_deg"],
                    "standoff_m": rb["standoff_m"],
                },
                "C_PaperA+B_SearchK16": {
                    "q_traj": best_c["roll"]["q_traj"].astype(np.float32),
                    "palm_traj": best_c["roll"]["palm_traj"].astype(np.float32),
                    "err_cm": best_c["final_err_cm"],
                    "valid": best_c["feas"]["r_valid"],
                    "ext": best_c["feas"]["ext_ratio_final"],
                    "lean_deg": best_c["spec"]["lean_deg"],
                    "standoff_m": best_c["spec"]["standoff_m"],
                },
                "D_Route2_协商规律": {
                    "q_traj": best_d["roll"]["q_traj"].astype(np.float32),
                    "palm_traj": best_d["roll"]["palm_traj"].astype(np.float32),
                    "err_cm": best_d["final_err_cm"],
                    "valid": best_d["feas"]["r_valid"],
                    "ext": best_d["feas"]["ext_ratio_final"],
                    "lean_deg": best_d["spec"]["lean_deg"],
                    "standoff_m": best_d["spec"]["standoff_m"],
                },
            },
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(test_s):
            print(f"      evaluated {i + 1}/{len(test_s)}", flush=True)

    vis_path = args.work_dir / "route2_vis.npy"
    np.save(vis_path, np.array(vis_records, dtype=object), allow_pickle=True)

    summary = [
        _summarize(res_a, "A_Naive_Delayed_0deg"),
        _summarize(res_b, "B_PaperA_LSTM_ProMP_0deg"),
        _summarize(res_c, "C_PaperA+B_SearchK16"),
        _summarize(res_d, "D_Route2_Plus_NegotLaw"),
    ]
    report = {
        "lstm_val_metrics": lstm_metrics,
        "test_causal_pred_cm": round(float(np.mean(test_pred_errs)), 2),
        "test_nopred_cm": round(float(np.mean(test_nopred_errs)), 2),
        "ablation": summary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json = args.work_dir / "route2_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== [5/5] Route 2 Ablation Summary ===")
    hdr = "%-26s %7s %7s %7s %6s %7s %7s %8s" % (
        "Condition", "ErrMed", "Succ<=5", "Valid%", "ExtMed", "InBand%", "Jerk", "ConcMed"
    )
    print(hdr)
    for row in summary:
        print("%-26s %6.2fcm %6.1f%% %6.1f%% %6.3f %6.1f%% %6.3f %7.2fcm" % (
            row["condition"], row["palm_err_med_cm"], row["success_le5cm_pct"],
            row["r_valid_pct"], row["ext_ratio_med"], row["ext_in_comfort_pct"],
            row["jerk_med_cm"], row["concession_med_cm"],
        ))
    print(f"Saved report -> {out_json} ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
