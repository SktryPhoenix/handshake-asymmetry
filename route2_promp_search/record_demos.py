"""Record Viser (.viser) 3D scene recordings and 4-way side-by-side MP4/GIF/PNG demos.

Generates for each selected test sequence:
1. `<seq>_D_route2.viser`: Native Viser 3D interactive recording via `get_scene_serializer()`.
2. `<seq>_compare4.mp4` & `<seq>_compare4.gif`: 30 fps 4-panel side-by-side animation
   (A_Naive_0deg | B_PaperA | C_SearchK16 | D_Route2_NegotLaw) showing G1, orange partner
   skeleton, and the 4 target spheres (Green=True, Yellow=LSTM, Red=Blend, Purple=G1).
3. `<seq>_contact_frame.png`: High-res final contact frame comparison strip.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import viser
import yourdfpy
from viser.extras import ViserUrdf

from config import DEFAULT_WORK_DIR, G1_URDF_PATH, G1_XML_PATH

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]
BONES = [(p, i) for i, p in enumerate(PARENTS) if p >= 0]

DEFAULT_SEQS = [
    "G043T000A001R012",
    "G038T000A001R007",
    "G038T000A001R006",
    "G023T000A001R008",
]

SHORT_TITLES = {
    "A_朴素延迟_0度": "A: Naive Delayed (0 deg)",
    "B_PaperA_LSTM+ProMP_0度": "B: Paper A (LSTM+ProMP, 0 deg)",
    "C_PaperA+B_SearchK16": "C: Paper A+B (Search K=16)",
    "D_Route2_协商规律": "D: Route 2 + Negotiation Law",
}


def quat_from_z_to(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ d)
    if c > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    half = 0.5 * np.arccos(c)
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def build_render_model(xml_path: Path):
    """Add mocap bodies for partner 22-joint skeleton and 4 target spheres to G1 MJCF."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    wb = spec.worldbody

    for i in range(22):
        b = wb.add_body(name=f"pjt_{i}")
        b.mocap = True
        b.pos = [0.0, 0.0, 1.0]
        g = b.add_geom()
        g.name = f"pjt_g_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [0.020, 0.0, 0.0]
        g.rgba = [0.96, 0.55, 0.18, 1.0]
        g.contype = 0
        g.conaffinity = 0

    for k in range(len(BONES)):
        b = wb.add_body(name=f"pb_{k}")
        b.mocap = True
        b.pos = [0.0, 0.0, 1.0]
        g = b.add_geom()
        g.name = f"pb_g_{k}"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [0.013, 0.08, 0.0]
        g.rgba = [0.96, 0.55, 0.18, 0.88]
        g.contype = 0
        g.conaffinity = 0

    # 4 indicator spheres matching view_viser.py colors
    spheres = [
        ("sp_true", [0.20, 0.88, 0.35, 0.95], 0.030),   # Green: partner true palm
        ("sp_lstm", [1.00, 0.86, 0.15, 0.85], 0.025),   # Yellow: LSTM predicted final
        ("sp_blend", [1.00, 0.25, 0.25, 0.90], 0.026),  # Red: Sigmoid blended target
        ("sp_g1", [0.72, 0.32, 1.00, 0.95], 0.028),     # Purple: G1 actual palm
    ]
    for sname, rgba, rad in spheres:
        b = wb.add_body(name=sname)
        b.mocap = True
        b.pos = [0.0, 0.0, 1.0]
        g = b.add_geom()
        g.name = f"{sname}_g"
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [rad, 0.0, 0.0]
        g.rgba = rgba
        g.contype = 0
        g.conaffinity = 0

    return spec.compile()


def export_viser_recording(rec: dict, urdf_path: Path, out_viser: Path,
                           scheme_key: str = "D_Route2_协商规律") -> None:
    """Serialize an offline interactive .viser 3D recording using ViserServer."""
    srv = viser.ViserServer(host="127.0.0.1", port=0)
    try:
        srv.scene.add_grid("/grid", width=4.0, height=4.0, position=(0.0, 0.0, 0.0))
        p2_bones = [
            srv.scene.add_line_segments(
                f"/p2/bone_{i}",
                points=np.zeros((1, 2, 3)),
                colors=np.array([[(255, 145, 50), (255, 145, 50)]], dtype=np.uint8),
                line_width=3.5,
            )
            for i in range(len(BONES))
        ]
        p2_joints = [
            srv.scene.add_icosphere(f"/p2/joint_{i}", radius=0.018, color=(255, 145, 50))
            for i in range(22)
        ]
        robot_root = srv.scene.add_frame("/g1_robot", show_axes=False)
        urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=True, build_scene_graph=True)
        v_robot = ViserUrdf(srv, urdf_or_path=urdf, root_node_name="/g1_robot")
        v_robot.show_visual = True

        sp_true = srv.scene.add_icosphere("/pts/partner_palm", radius=0.032, color=(50, 220, 90))
        sp_lstm = srv.scene.add_icosphere("/pts/lstm_pred", radius=0.028, color=(255, 220, 40))
        sp_blend = srv.scene.add_icosphere("/pts/blend_target", radius=0.028, color=(255, 65, 65))
        sp_g1 = srv.scene.add_icosphere("/pts/g1_palm", radius=0.030, color=(180, 80, 255))

        ser = srv.get_scene_serializer()
        sc = rec["schemes"][scheme_key]
        q_traj = sc["q_traj"]
        p2 = rec["j2_full"]
        n_frames = len(p2)

        for t in range(n_frames):
            pts = p2[t]
            for i, (a, b) in enumerate(BONES):
                p2_bones[i].points = np.stack([pts[a], pts[b]])[None]
            for i, h in enumerate(p2_joints):
                h.position = tuple(float(v) for v in pts[i])

            q_t = q_traj[min(t, len(q_traj) - 1)]
            robot_root.position = (float(q_t[0]), float(q_t[1]), float(q_t[2]))
            robot_root.wxyz = tuple(float(v) for v in q_t[3:7])
            v_robot.update_cfg(np.asarray(q_t[7:36], dtype=float))

            sp_true.position = tuple(float(v) for v in rec["partner_palm"][t])
            sp_lstm.position = tuple(float(v) for v in rec["lstm_pred"][t])
            sp_blend.position = tuple(float(v) for v in rec["blend_target"][t])
            sp_g1.position = tuple(float(v) for v in sc["palm_traj"][min(t, len(sc["palm_traj"]) - 1)])
            ser.insert_sleep(1.0 / 30.0)

        out_viser.write_bytes(ser.serialize())
    finally:
        srv.stop()


def render_4way_video(rec: dict, m: mujoco.MjModel, renderer: mujoco.Renderer,
                      out_dir: Path, width: int = 360, height: int = 340) -> dict:
    """Render 4-panel side-by-side MP4, GIF, and final contact PNG for one sequence."""
    d = mujoco.MjData(m)
    p2 = rec["j2_full"].astype(float)
    n_frames = len(p2)

    pjt_mids = [int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"pjt_{i}")]) for i in range(22)]
    pb_mids = [int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"pb_{k}")]) for k in range(len(BONES))]
    sp_mids = {
        k: int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k)])
        for k in ("sp_true", "sp_lstm", "sp_blend", "sp_g1")
    }

    bonelen = np.array([np.linalg.norm(p2[0, b] - p2[0, a]) for a, b in BONES])
    for k, bl in enumerate(bonelen):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"pb_g_{k}")
        m.geom_size[gid][1] = max(0.015, bl / 2.0)

    scheme_keys = list(rec["schemes"].keys())
    # Camera fixed per sequence looking perpendicular to the interaction line at contact
    q_ref_end = rec["schemes"][scheme_keys[-1]]["q_traj"][-1]
    base_xy = q_ref_end[:2]
    part_xy = p2[-1, 0, :2]
    line = part_xy - base_xy
    ang = float(np.degrees(np.arctan2(line[1], line[0])))

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = ang + 88.0
    cam.elevation = -14.0
    cam.distance = 2.05
    cam.lookat = np.array([
        0.5 * float(base_xy[0] + part_xy[0]),
        0.5 * float(base_xy[1] + part_xy[1]),
        0.88,
    ])

    frames_combined = []
    for t in range(n_frames):
        panels = []
        for skey in scheme_keys:
            sc = rec["schemes"][skey]
            q_t = sc["q_traj"][min(t, len(sc["q_traj"]) - 1)]
            p_g1 = sc["palm_traj"][min(t, len(sc["palm_traj"]) - 1)]
            p_true = rec["partner_palm"][t]
            p_lstm = rec["lstm_pred"][t]
            p_blend = rec["blend_target"][t]

            d.qpos[:36] = q_t[:36]
            for i in range(22):
                d.mocap_pos[pjt_mids[i]] = p2[t, i]
                d.mocap_quat[pjt_mids[i]] = [1.0, 0.0, 0.0, 0.0]
            for k, (a, b) in enumerate(BONES):
                d.mocap_pos[pb_mids[k]] = 0.5 * (p2[t, a] + p2[t, b])
                d.mocap_quat[pb_mids[k]] = quat_from_z_to(p2[t, b] - p2[t, a])

            d.mocap_pos[sp_mids["sp_true"]] = p_true
            d.mocap_pos[sp_mids["sp_lstm"]] = p_lstm
            d.mocap_pos[sp_mids["sp_blend"]] = p_blend
            d.mocap_pos[sp_mids["sp_g1"]] = p_g1

            mujoco.mj_forward(m, d)
            renderer.update_scene(d, cam)
            img = renderer.render().copy()

            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, width - 1, 38], fill=(18, 22, 30))
            title = SHORT_TITLES.get(skey, skey)
            dist_now = float(np.linalg.norm(p_g1 - p_true) * 100.0)
            dr.text((8, 4), title, fill=(240, 245, 255))
            dr.text(
                (8, 20),
                f"f={t+1}/{n_frames} | d={dist_now:.1f}cm (final {sc['err_cm']:.1f}cm) | lean={sc['lean_deg']:.0f}deg | ext={sc['ext']:.2f}",
                fill=(170, 230, 185) if sc["err_cm"] <= 5.0 else (255, 185, 140),
            )
            panels.append(np.asarray(im))

        row = np.concatenate(panels, axis=1)
        frames_combined.append(row)

    # Hold the final contact frame for 12 extra frames (0.4s) so the viewer sees the clasp clearly
    for _ in range(12):
        frames_combined.append(frames_combined[-1])

    seq = rec["seq"]
    mp4_path = out_dir / f"{seq}_compare4.mp4"
    gif_path = out_dir / f"{seq}_compare4.gif"
    png_path = out_dir / f"{seq}_contact_frame.png"

    pil_frames = [Image.fromarray(f) for f in frames_combined]
    pil_frames[-1].save(png_path)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / 25),
        loop=0,
        optimize=True,
    )
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(gif_path),
                "-movflags", "faststart", "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4_path),
            ],
            check=False,
        )
    return {
        "seq": seq,
        "mp4": str(mp4_path),
        "gif": str(gif_path),
        "png": str(png_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", type=Path, default=DEFAULT_WORK_DIR / "route2_vis.npy")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_WORK_DIR / "demos")
    ap.add_argument("--seqs", default=",".join(DEFAULT_SEQS))
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--height", type=int, default=340)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = list(np.load(args.vis, allow_pickle=True))
    by_seq = {r["seq"]: r for r in records}
    target_seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]

    m = build_render_model(G1_XML_PATH)
    renderer = mujoco.Renderer(m, height=args.height, width=args.width)

    for seq in target_seqs:
        if seq not in by_seq:
            print(f"[warn] {seq} not in vis records, skipping")
            continue
        rec = by_seq[seq]
        print(f"[demo] rendering {seq} ({rec['hand_pair']}, {len(rec['j2_full'])} frames)...", flush=True)
        info = render_4way_video(rec, m, renderer, args.out_dir, width=args.width, height=args.height)
        viser_path = args.out_dir / f"{seq}_D_route2.viser"
        export_viser_recording(rec, G1_URDF_PATH, viser_path)
        print(f"       -> {info['gif']} | {info['mp4']} | {viser_path.name}", flush=True)


if __name__ == "__main__":
    main()
