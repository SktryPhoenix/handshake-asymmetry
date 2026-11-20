"""Optional bridge to ~/repos/GMR (runs in the `gmr` conda env via subprocess).

Use this when you want full-body GMR retargeting (like Paper B 2610.15274's primary
retargeter) on a specific Inter-X SMPL-X/SMPL-H sequence.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from config import GMR_PYTHON, GMR_REPO_DIR


def retarget_with_gmr(npz_path: Path, out_pkl: Path, robot: str = "unitree_g1") -> None:
    """Invoke GMR's smplx_to_robot.py inside the gmr conda environment."""
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(GMR_PYTHON),
        str(GMR_REPO_DIR / "scripts" / "smplx_to_robot.py"),
        "--smplx_file", str(npz_path),
        "--robot", robot,
        "--save_path", str(out_pkl),
    ]
    print("[gmr] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(GMR_REPO_DIR), check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--robot", default="unitree_g1")
    args = ap.parse_args()
    retarget_with_gmr(args.npz, args.out, robot=args.robot)
