"""Step 5 (Layer R): Embodiment-Aware Reaction Search + Empirical Negotiation Prior.

Combines:
1. Paper B (2610.15274 Eq. 5-12):
   - Sample K candidate reactions from the learned ProMP distribution + posture variations.
   - Retarget each candidate to Unitree G1 and check eligibility E_t (human validity,
     robot validity r_{t,k}, semantic handshake height/side floor).
   - Score candidates by standardized kinematic benefit B_kin and interaction benefit B_int.
2. Our Negotiation Law Prior (research/law/):
   - Incorporates empirical forward lean (~25 deg), standoff (~84 cm), horizontal midpoint
     (lambda=0.5), height concession (-0.22 * dh), and comfort extension band [0.84, 0.90].
   - Explicitly computes residual concession_cm needed from the human partner when G1
     reaches its physical arm envelope.
"""

from __future__ import annotations

import numpy as np

from config import (
    G1_ARM_REACH_M,
    LAW_CLASP_HEIGHT_M,
    LAW_EXT_TARGET_HIGH,
    LAW_EXT_TARGET_LOW,
    LAW_LEAN_DEG,
    LAW_STANDOFF_M,
    LAW_Z_CONCESSION_SLOPE,
)
from g1_kinematics import G1Kinematics
from promp import ProMP


def negotiate_clasp_target(
    raw_target_palms: np.ndarray,
    h_human_m: float,
    h_robot_m: float = 1.320,
    use_law_z: bool = True,
) -> np.ndarray:
    """Apply empirical vertical concession law (delta_z ~= -0.22 * delta_h) toward ~100 cm."""
    if not use_law_z:
        return raw_target_palms
    targets = raw_target_palms.copy()
    # At the final approach portion, nudge clasp z subtly toward empirical social norm
    dh = h_human_m - h_robot_m  # positive since human > G1
    desired_z = np.clip(LAW_CLASP_HEIGHT_M + LAW_Z_CONCESSION_SLOPE * dh, 0.88, 1.05)
    # Blend smoothly only in the last 40% of reaching so early trajectory isn't distorted
    t_len = len(targets)
    w = np.clip(np.linspace(-0.5, 0.6, t_len), 0.0, 0.6)
    targets[:, 2] = (1.0 - w) * targets[:, 2] + w * desired_z
    return targets


def _candidate_specs(
    promp: ProMP,
    t_len: int,
    k: int,
    rng: np.random.Generator,
    use_negotiation_pool: bool,
) -> list[dict]:
    """Generate K candidate reactions: Candidate 0 is always the mean baseline."""
    arm_samples = promp.sample(k=k, t_len=t_len, rng=rng, include_mean=True, scale=1.0)
    specs = []
    for idx in range(k):
        if idx == 0:
            lean = LAW_LEAN_DEG if use_negotiation_pool else 0.0
            standoff = LAW_STANDOFF_M
        elif use_negotiation_pool:
            lean = float(np.clip(rng.normal(LAW_LEAN_DEG, 4.0), 10.0, 29.0))
            standoff = float(np.clip(rng.normal(LAW_STANDOFF_M, 0.04), 0.74, 0.92))
        else:
            lean = float(np.clip(rng.uniform(0.0, 28.0), 0.0, 29.0))
            standoff = float(np.clip(rng.normal(0.84, 0.05), 0.72, 0.95))
        specs.append({
            "k": idx,
            "arm_4dof": arm_samples[idx],
            "lean_deg": lean,
            "standoff_m": standoff,
        })
    return specs


def search_best_reaction(
    g1: G1Kinematics,
    promp: ProMP,
    partner_pelvis: np.ndarray,
    robot_pelvis: np.ndarray,
    partner_palm_true: np.ndarray,
    target_palms: np.ndarray,
    side: str = "R",
    k_candidates: int = 16,
    use_negotiation_prior: bool = True,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run Paper B inference-time search over K ProMP + posture candidates."""
    rng = rng or np.random.default_rng(0)
    t_len = len(target_palms)
    specs = _candidate_specs(promp, t_len, k_candidates, rng, use_negotiation_prior)

    evaluated = []
    for sp in specs:
        arm_traj = sp["arm_4dof"].copy()
        if side == "L":
            arm_traj[:, 1] *= -1.0
            arm_traj[:, 2] *= -1.0
        roll = g1.rollout_trajectory(
            partner_pelvis=partner_pelvis,
            robot_pelvis=robot_pelvis,
            target_palms=target_palms,
            arm_4dof_traj=arm_traj,
            side=side,
            standoff_m=sp["standoff_m"],
            lean_deg=sp["lean_deg"],
        )
        feas = g1.evaluate_feasibility(roll["q_traj"], roll["palm_traj"], roll["sh_traj"])

        # Final palm reach error wrt partner's true palm at contact
        final_err_m = float(np.linalg.norm(roll["palm_traj"][-1] - partner_palm_true[-1]))
        # Target tracking error along trajectory
        mean_track_err_m = float(np.mean(roll["err_traj"]))
        # Extension deviation from human comfort band [0.84, 0.90]
        ext = feas["ext_ratio_final"]
        ext_band_pen = max(0.0, LAW_EXT_TARGET_LOW - ext, ext - LAW_EXT_TARGET_HIGH)

        # Residual concession needed from human if G1 hand still falls short (>3cm)
        sh_to_target = float(np.linalg.norm(partner_palm_true[-1] - roll["sh_traj"][-1]))
        concession_m = max(0.0, sh_to_target - G1_ARM_REACH_M)

        # Semantic handshake floor: palm height at contact within [0.70, 1.15] m
        palm_z_final = float(roll["palm_traj"][-1, 2])
        sem_ok = 0.70 <= palm_z_final <= 1.15

        # Paper B costs: J_kin (robot feasibility/smoothness) and J_int (interaction fidelity)
        j_kin = (
            (mean_track_err_m / 0.10)
            + (feas["jerk_cm"] / 2.0)
            + (feas["max_self_pen_m"] / 0.03)
            + (feas["max_floor_pen_m"] / 0.03)
        )
        if use_negotiation_prior:
            j_kin += 2.0 * (ext_band_pen / 0.10)

        j_int = (final_err_m / 0.05) + (concession_m / 0.05)

        evaluated.append({
            "spec": sp,
            "roll": roll,
            "feas": feas,
            "sem_ok": sem_ok,
            "j_kin": float(j_kin),
            "j_int": float(j_int),
            "final_err_cm": final_err_m * 100.0,
            "concession_cm": concession_m * 100.0,
        })

    # Eligible set E_t (Eq. 10): robot valid & semantic handshake height valid
    eligible = [e for e in evaluated if e["feas"]["r_valid"] and e["sem_ok"]]
    if not eligible:
        eligible = [evaluated[0]]

    # Standardized benefit relative to candidate 0 (Eq. 8-9, 12)
    j_kin_0 = evaluated[0]["j_kin"]
    j_int_0 = evaluated[0]["j_int"]
    gamma_kin = max(float(np.std([e["j_kin"] for e in evaluated])), 1e-4)
    gamma_int = max(float(np.std([e["j_int"] for e in evaluated])), 1e-4)

    best = eligible[0]
    best_score = -1e9
    for e in eligible:
        b_kin = (j_kin_0 - e["j_kin"]) / gamma_kin
        b_int = (j_int_0 - e["j_int"]) / gamma_int
        score = 0.5 * (b_kin + b_int)
        if score > best_score:
            best_score = score
            best = e

    best["k_selected"] = best["spec"]["k"]
    best["eligible_count"] = len(eligible)
    best["baseline_0"] = evaluated[0]
    return best
