"""Step 4 (Layer R/D): Unitree G1 MuJoCo Kinematics, Anatomical IK, and Feasibility Checks.

Fixes applied after Viser visual inspection:
- Enforces strict anatomical handshake joint bounds (e.g. elbow <= +1.40 rad to prevent
  backward hyperextension / "anti-joint", waist_yaw in [-0.35, +0.35], shoulder_pitch <= +0.30).
- Fixes human 4-DoF -> G1 elbow sign mapping: G1 elbow=0 is 90deg bent and +pi/2 is straight,
  so g1_elbow = pi/2 - human_el_flex.
- Locks wrist joints near 0 (neutral palm-inward handshake orientation) so position-only IK
  cannot spin wrists into unnatural pretzel angles.
- Smoothly ramps arm motion from a natural hanging ready pose (no frame-0 jump) and enforces
  anatomical validity inside evaluate_feasibility().
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from config import (
    DELTA_FLOOR_M,
    DELTA_LIMIT_RAD,
    DELTA_SELF_M,
    G1_ARM_REACH_M,
    G1_XML_PATH,
)

# Anatomical handshake bounds (rad), much tighter than mechanical motor stops
# Keeps the elbow naturally pointing downward (prevents inward roll + outward yaw elbow flip)
ANAT_LIMITS = {
    "waist_yaw_joint": (-0.35, 0.35),
    "waist_roll_joint": (-0.15, 0.15),
    "waist_pitch_joint": (0.0, 0.50),
    "L_shoulder_pitch": (-1.65, 0.05),
    "L_shoulder_roll": (-0.05, 0.75),
    "L_shoulder_yaw": (-0.45, 0.20),
    "L_elbow": (-0.90, 1.18),       # keeps natural elbow bend; >1.45 hyperextends backward!
    "R_shoulder_pitch": (-1.65, 0.05),
    "R_shoulder_roll": (-0.75, 0.05),
    "R_shoulder_yaw": (-0.20, 0.45),
    "R_elbow": (-0.90, 1.18),       # keeps natural elbow bend; >1.45 hyperextends backward!
}


class G1Kinematics:
    """MuJoCo wrapper for G1 handshake kinematics and anatomical feasibility."""

    def __init__(self, xml_path: Path = G1_XML_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.arm_joints = {}
        self.ik_joints = {}
        self.hand_body = {}
        self.shoulder_body = {}
        for side, prefix in (("L", "left"), ("R", "right")):
            jnames = [
                f"{prefix}_shoulder_pitch_joint",
                f"{prefix}_shoulder_roll_joint",
                f"{prefix}_shoulder_yaw_joint",
                f"{prefix}_elbow_joint",
                f"{prefix}_wrist_roll_joint",
                f"{prefix}_wrist_pitch_joint",
                f"{prefix}_wrist_yaw_joint",
            ]
            qpos_ids = [int(self.model.joint(n).qposadr[0]) for n in jnames]
            dof_ids = [int(self.model.joint(n).dofadr[0]) for n in jnames]
            j_ids = [int(self.model.joint(n).id) for n in jnames]
            self.arm_joints[side] = (qpos_ids, dof_ids, j_ids)

            # Active 4-DoF arm joints (wrists kept at neutral handshake orientation 0)
            ik_names = jnames[:4]
            self.ik_joints[side] = (
                qpos_ids[:4],
                dof_ids[:4],
                j_ids[:4],
                [
                    ANAT_LIMITS[f"{side}_shoulder_pitch"],
                    ANAT_LIMITS[f"{side}_shoulder_roll"],
                    ANAT_LIMITS[f"{side}_shoulder_yaw"],
                    ANAT_LIMITS[f"{side}_elbow"],
                ],
            )
            self.hand_body[side] = int(self.model.body(f"{prefix}_rubber_hand_link").id)
            self.shoulder_body[side] = int(self.model.body(f"{prefix}_shoulder_pitch_link").id)

        self.waist_qpos = {
            k: int(self.model.joint(f"waist_{k}_joint").qposadr[0])
            for k in ("yaw", "roll", "pitch")
        }
        self.waist_dof = {
            k: int(self.model.joint(f"waist_{k}_joint").dofadr[0])
            for k in ("yaw", "roll", "pitch")
        }
        self.waist_jnt = {
            k: int(self.model.joint(f"waist_{k}_joint").id)
            for k in ("yaw", "roll", "pitch")
        }
        self.foot_bodies = [
            i for i in range(self.model.nbody)
            if "ankle_roll_sphere" in self.model.body(i).name
        ]
        self.stand_pelvis_z = self._compute_stand_pelvis_z()

    def _compute_stand_pelvis_z(self) -> float:
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0
        mujoco.mj_kinematics(self.model, self.data)
        min_z = min(float(self.data.body(b).xpos[2]) for b in self.foot_bodies)
        return -min_z

    def neutral_qpos(self, pelvis_xy: np.ndarray, yaw_rad: float,
                     lean_deg: float = 0.0) -> np.ndarray:
        """Create grounded G1 qpos facing yaw_rad with relaxed hanging arms."""
        q = np.zeros(self.model.nq, dtype=np.float64)
        q[0], q[1], q[2] = float(pelvis_xy[0]), float(pelvis_xy[1]), self.stand_pelvis_z
        half = 0.5 * yaw_rad
        q[3], q[4], q[5], q[6] = np.cos(half), 0.0, 0.0, np.sin(half)
        lean_rad = float(np.clip(np.radians(lean_deg), 0.0, 0.50))
        q[self.waist_qpos["pitch"]] = lean_rad
        # Relaxed hanging arm posture (G1 elbow=+1.05 is naturally relaxed without hyperextension)
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            q_ids = self.arm_joints[side][0]
            q[q_ids[0]] = -0.05
            q[q_ids[1]] = 0.18 * sgn
            q[q_ids[2]] = 0.0
            q[q_ids[3]] = 1.05
        return q

    def arm_prior_from_4dof(self, arm_4dof: np.ndarray, side: str = "R") -> np.ndarray:
        """Map human 4-DoF (sh_pitch, sh_roll, sh_yaw, el_flex) to G1 4-DoF anatomical angles."""
        sp, sr, sy, el = [float(v) for v in arm_4dof]
        limits = self.ik_joints[side][3]
        # Human sh_pitch > 0 lifts forward; G1 shoulder_pitch < 0 lifts forward
        g1_sp = float(np.clip(-sp, *limits[0]))
        g1_sr = float(np.clip(sr, *limits[1]))
        g1_sy = float(np.clip(0.50 * sy, *limits[2]))
        # G1 elbow=pi/2 is straight, 0 is 90deg flex, <0 is deep flex
        g1_el = float(np.clip((0.5 * np.pi) - el, *limits[3]))
        return np.array([g1_sp, g1_sr, g1_sy, g1_el], dtype=np.float64)

    def g1_to_human_4dof(self, g1_arm4: np.ndarray) -> np.ndarray:
        """Inverse of arm_prior_from_4dof for conditioning ProMP in human 4-DoF space."""
        sp = -float(g1_arm4[0])
        sr = float(g1_arm4[1])
        sy = float(g1_arm4[2]) / 0.50
        el = (0.5 * np.pi) - float(g1_arm4[3])
        return np.array([sp, sr, sy, el], dtype=np.float32)

    def palm_and_shoulder(self, qpos: np.ndarray, side: str = "R") -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        palm = self.data.body(self.hand_body[side]).xpos.copy()
        sh = self.data.body(self.shoulder_body[side]).xpos.copy()
        return palm, sh

    def solve_arm_ik_step(
        self,
        q_init: np.ndarray,
        target_palm: np.ndarray,
        side: str = "R",
        q_arm4_prior: np.ndarray | None = None,
        iters: int = 6,
        damping: float = 0.10,
        null_gain: float = 0.18,
        allow_waist_yaw: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """DLS IK over anatomical (waist_yaw + 4-DoF arm); wrists stay neutral at 0."""
        q = q_init.copy()
        q_ids, d_ids, _, limits = self.ik_joints[side]
        q_ids = list(q_ids)
        d_ids = list(d_ids)
        bounds = list(limits)

        if allow_waist_yaw:
            q_ids = [self.waist_qpos["yaw"]] + q_ids
            d_ids = [self.waist_dof["yaw"]] + d_ids
            bounds = [ANAT_LIMITS["waist_yaw_joint"]] + bounds

        if q_arm4_prior is not None:
            q_prior_full = np.concatenate(([0.0], q_arm4_prior)) if allow_waist_yaw else q_arm4_prior
        else:
            q_prior_full = q[q_ids].copy()

        body_id = self.hand_body[side]
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)

        for _ in range(iters):
            self.data.qpos[:] = q
            mujoco.mj_kinematics(self.model, self.data)
            curr = self.data.body(body_id).xpos
            err = target_palm - curr
            if np.linalg.norm(err) < 3e-3:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp, None, body_id)
            j_sub = jacp[:, d_ids]
            jjt = j_sub @ j_sub.T + (damping ** 2) * eye3
            j_pinv = j_sub.T @ np.linalg.inv(jjt)
            dq_task = j_pinv @ err
            null_proj = np.eye(len(q_ids)) - j_pinv @ j_sub
            dq_null = null_gain * (q_prior_full - q[q_ids])
            dq = np.clip(dq_task + null_proj @ dq_null, -0.15, 0.15)
            q[q_ids] += dq
            for idx, (lo, hi) in zip(q_ids, bounds):
                q[idx] = float(np.clip(q[idx], lo, hi))

        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        palm = self.data.body(body_id).xpos.copy()
        return q, palm, float(np.linalg.norm(target_palm - palm))

    def solve_goal_arm4(
        self,
        robot_xy: np.ndarray,
        yaw_rad: float,
        target_palm_final: np.ndarray,
        arm_4dof_init: np.ndarray,
        side: str = "R",
        lean_deg: float = 25.0,
    ) -> np.ndarray:
        """Find anatomical 4-DoF G1 arm target to reach final clasp palm (used to condition ProMP)."""
        q = self.neutral_qpos(robot_xy, yaw_rad, lean_deg=lean_deg)
        q_arm_ids = self.ik_joints[side][0]
        # Start from natural mid-reach handshake pose to avoid hanging-arm singularity
        q[q_arm_ids] = np.array([-0.65, -0.10 if side == "R" else 0.10, 0.0, 0.20])
        q_prior = self.arm_prior_from_4dof(arm_4dof_init, side=side)
        for _ in range(3):
            q, _, _ = self.solve_arm_ik_step(
                q, target_palm_final, side=side, q_arm4_prior=q_prior, iters=6, allow_waist_yaw=True,
            )
        return q[q_arm_ids].copy()

    def rollout_trajectory(
        self,
        partner_pelvis: np.ndarray,
        robot_pelvis: np.ndarray,
        target_palms: np.ndarray,
        arm_4dof_traj: np.ndarray,
        side: str = "R",
        standoff_m: float = 0.84,
        lean_deg: float = 25.0,
        max_joint_step: float = 0.12,
    ) -> dict:
        """Roll out smooth, anatomically bounded G1 reaching trajectory.

        Follows the ProMP arm prior smoothly from ready pose at t=0 and blends in
        Cartesian palm tracking progressively toward the contact instant (frac -> 1.0),
        preventing frame-0 snapping or inverted joints.
        """
        t_len = len(target_palms)
        dir_xy = robot_pelvis[-1, :2] - partner_pelvis[-1, :2]
        norm_xy = float(np.linalg.norm(dir_xy))
        if norm_xy < 1e-4:
            dir_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            dir_xy = dir_xy / norm_xy

        robot_xy = partner_pelvis[-1, :2] + dir_xy * standoff_m
        yaw_rad = float(np.arctan2(-dir_xy[1], -dir_xy[0]))

        q_traj = np.zeros((t_len, self.model.nq), dtype=np.float64)
        palm_traj = np.zeros((t_len, 3), dtype=np.float64)
        sh_traj = np.zeros((t_len, 3), dtype=np.float64)
        err_traj = np.zeros(t_len, dtype=np.float64)

        q_prev = self.neutral_qpos(robot_xy, yaw_rad, lean_deg=0.0)
        q_arm_ids = self.ik_joints[side][0]
        start_palm, _ = self.palm_and_shoulder(q_prev, side=side)

        for t in range(t_len):
            frac = t / max(t_len - 1, 1)
            q_init = q_prev.copy()
            q_init[self.waist_qpos["pitch"]] = float(np.clip(np.radians(lean_deg) * frac, 0.0, 0.50))
            q_prior = self.arm_prior_from_4dof(arm_4dof_traj[t], side=side)

            # Smooth Cartesian guide from G1's natural start palm -> online target_palms[t]
            guide_w = float(frac ** 0.75)
            guide_palm = (1.0 - guide_w) * start_palm + guide_w * target_palms[t]

            q_sol, _, _ = self.solve_arm_ik_step(
                q_init, guide_palm, side=side, q_arm4_prior=q_prior,
                iters=5, null_gain=0.18 * (1.0 - 0.5 * frac), allow_waist_yaw=True,
            )
            # Enforce velocity cap from frame 0 relative to hanging ready pose
            delta = np.clip(q_sol[7:] - q_prev[7:], -max_joint_step, max_joint_step)
            q_sol[7:] = q_prev[7:] + delta

            palm_t, sh_t = self.palm_and_shoulder(q_sol, side=side)
            q_traj[t] = q_sol
            palm_traj[t] = palm_t
            sh_traj[t] = sh_t
            err_traj[t] = float(np.linalg.norm(guide_palm - palm_t))
            q_prev = q_sol

        return {
            "q_traj": q_traj,
            "palm_traj": palm_traj,
            "sh_traj": sh_traj,
            "err_traj": err_traj,
            "robot_xy": robot_xy,
            "yaw_rad": yaw_rad,
        }

    def _anatomical_violation(self, q: np.ndarray) -> float:
        """Check violation against ANAT_LIMITS across waist & both arms."""
        viol = 0.0
        for wkey in ("yaw", "roll", "pitch"):
            lo, hi = ANAT_LIMITS[f"waist_{wkey}_joint"]
            v = float(q[self.waist_qpos[wkey]])
            viol = max(viol, lo - v, v - hi)
        for side in ("L", "R"):
            q_ids, _, _, bounds = self.ik_joints[side]
            for idx, (lo, hi) in zip(q_ids, bounds):
                v = float(q[idx])
                viol = max(viol, lo - v, v - hi)
        return viol

    def evaluate_feasibility(self, q_traj: np.ndarray, palm_traj: np.ndarray,
                             sh_traj: np.ndarray) -> dict:
        """Compute robot validity r_{t,k} (self/floor collision + anatomical limits) & comfort."""
        max_self_pen = 0.0
        max_floor_pen = 0.0
        max_lim_viol = 0.0

        for t in range(len(q_traj)):
            self.data.qpos[:] = q_traj[t]
            mujoco.mj_fwdPosition(self.model, self.data)
            # Hardware + anatomical limit violation
            for jid in range(1, self.model.njnt):
                qidx = int(self.model.jnt_qposadr[jid])
                lo, hi = self.model.jnt_range[jid]
                val = q_traj[t, qidx]
                max_lim_viol = max(max_lim_viol, max(0.0, lo - val, val - hi))
            max_lim_viol = max(max_lim_viol, self._anatomical_violation(q_traj[t]))

            min_fz = min(float(self.data.body(b).xpos[2]) for b in self.foot_bodies)
            max_floor_pen = max(max_floor_pen, max(0.0, -min_fz))
            for c_idx in range(self.data.ncon):
                c = self.data.contact[c_idx]
                if c.geom1 == 0 or c.geom2 == 0:
                    continue
                if c.dist < 0:
                    max_self_pen = max(max_self_pen, float(-c.dist))

        is_valid = (
            max_self_pen <= DELTA_SELF_M
            and max_floor_pen <= DELTA_FLOOR_M
            and max_lim_viol <= DELTA_LIMIT_RAD
        )
        v = np.diff(palm_traj, axis=0)
        acc = np.diff(v, axis=0) if len(v) > 1 else np.zeros((1, 3))
        jerk_cm = float(np.median(np.linalg.norm(acc, axis=1)) * 100.0)
        ext_final = float(np.linalg.norm(palm_traj[-1] - sh_traj[-1]) / G1_ARM_REACH_M)

        return {
            "r_valid": bool(is_valid),
            "max_self_pen_m": round(max_self_pen, 4),
            "max_floor_pen_m": round(max_floor_pen, 4),
            "max_lim_viol_rad": round(max_lim_viol, 5),
            "jerk_cm": round(jerk_cm, 3),
            "ext_ratio_final": round(ext_final, 3),
        }
