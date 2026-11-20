"""Configuration and constants for Route 2 (ProMP + LSTM + Embodiment-Aware Search)."""

from __future__ import annotations

from pathlib import Path

# --- Server & local paths (3090 authoritative paths) ---
INTERX_A001_DIR = Path("/mnt/sdc/ltr/02-paper-repro-and-data/data/datasets/interx_a001")
SMPLX_MODEL_PATH = Path("/mnt/sdc/ltr/models/smplx/SMPLX_NEUTRAL.npz")
G1_XML_PATH = Path(
    "/mnt/sdc/ltr/repos/OmniRetarget/src/holosoma_retargeting/"
    "holosoma_retargeting/models/g1/g1_29dof.xml"
)
G1_URDF_PATH = Path(
    "/mnt/sdc/ltr/repos/OmniRetarget/src/holosoma_retargeting/"
    "holosoma_retargeting/models/g1/g1_29dof.urdf"
)
GMR_REPO_DIR = Path("/mnt/sdc/ltr/repos/GMR")
GMR_PYTHON = Path("/mnt/sdc/ltr/miniconda3/envs/gmr/bin/python")
HS_PYTHON = Path("/mnt/sdc/ltr/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python3")

DEFAULT_WORK_DIR = Path("/mnt/sdc/ltr/runs/handshake_route2")

# --- SMPL-X 55-joint indices (first 52 match SMPL-H) ---
PELVIS = 0
ANKLE_L, ANKLE_R = 7, 8
TOE_L, TOE_R = 10, 11
SPINE1, SPINE2, SPINE3 = 3, 6, 9
NECK, HEAD = 12, 15
COLLAR_L, COLLAR_R = 13, 14
SHOULDER_L, SHOULDER_R = 16, 17
ELBOW_L, ELBOW_R = 18, 19
WRIST_L, WRIST_R = 20, 21
MCP_L, MCP_R = 28, 43
FEET_JOINTS = [ANKLE_L, ANKLE_R, TOE_L, TOE_R]

# 14 symmetric upper-body joints (+ active palm = 15 points, matching Hand Reaching 2103.00616)
UPPER_BODY_14 = [
    PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD,
    COLLAR_L, SHOULDER_L, ELBOW_L, WRIST_L,
    COLLAR_R, SHOULDER_R, ELBOW_R, WRIST_R,
]

# --- G1 & Human empirical negotiation constants (from research/law/) ---
G1_ARM_REACH_M = 0.3873          # shoulder -> wrist (0.2031 + 0.1843)
HUMAN_ARM_REACH_M = 0.5207       # SMPL-X neutral
G1_SHOULDER_H_M = 1.0848
HUMAN_SHOULDER_H_M = 1.3140

LAW_LEAN_DEG = 25.0              # human empirical trunk pitch [22, 28] deg
LAW_STANDOFF_M = 0.84            # empirical pelvis-to-pelvis distance [0.80, 0.88] m
LAW_CLASP_HEIGHT_M = 1.00        # empirical social constant ~100 cm
LAW_Z_CONCESSION_SLOPE = -0.22   # delta_z_clasp ~= -0.22 * delta_h
LAW_EXT_TARGET_LOW = 0.84        # empirical extension ratio comfort band
LAW_EXT_TARGET_HIGH = 0.90

# --- Paper B (2610.15274) feasibility thresholds (Eq. 11) ---
DELTA_SELF_M = 0.03
DELTA_FLOOR_M = 0.03
DELTA_LIMIT_RAD = 1e-4

# --- Timing & contact defaults ---
FPS = 30.0
CONTACT_DIST_M = 0.10
MIN_CONTACT_RATIO = 0.05
HISTORY_FRAMES = 10
