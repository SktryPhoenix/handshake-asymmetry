"""Interactive Viser 3D Viewer for Route 2 (A/B/C/D ablation + LSTM prediction + human skeleton).

Usage on 3090:
    python3 view_viser.py --vis ~/runs/handshake_route2/route2_vis.npy --port 8091
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from config import DEFAULT_WORK_DIR, G1_URDF_PATH

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]
BONES = np.array([(p, i) for i, p in enumerate(PARENTS) if p >= 0], dtype=int)


class SkeletonVis:
    """22-joint SMPL-X skeleton rendered in viser."""

    def __init__(self, server: viser.ViserServer, prefix: str,
                 rgb: tuple[int, int, int], radius: float = 0.018) -> None:
        self.bones = [
            server.scene.add_line_segments(
                f"/{prefix}/bone_{i}",
                points=np.zeros((1, 2, 3)),
                colors=np.array([[rgb, rgb]], dtype=np.uint8),
                line_width=3.5,
            )
            for i in range(len(BONES))
        ]
        self.joints = [
            server.scene.add_icosphere(
                f"/{prefix}/joint_{i}", radius=radius, color=rgb, position=(0, 0, 0),
            )
            for i in range(len(PARENTS))
        ]

    def update(self, pts: np.ndarray | None, visible: bool) -> None:
        show = visible and pts is not None
        for i, (a, b) in enumerate(BONES):
            self.bones[i].visible = show
            if show:
                self.bones[i].points = np.stack([pts[a], pts[b]])[None]
        for i, h in enumerate(self.joints):
            h.visible = show
            if show:
                h.position = tuple(float(v) for v in pts[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", type=Path, default=DEFAULT_WORK_DIR / "route2_vis.npy")
    ap.add_argument("--urdf", type=Path, default=G1_URDF_PATH)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    records = list(np.load(args.vis, allow_pickle=True))
    if not records:
        raise SystemExit(f"Empty vis file: {args.vis}")

    seq_names = [f"{r['seq']} ({r['hand_pair']})" for r in records]
    scheme_names = list(records[0]["schemes"].keys())

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/grid", width=5.0, height=5.0, position=(0.0, 0.0, 0.0))

    sk_partner = SkeletonVis(server, "partner_p2", (255, 145, 50))
    sk_orig = SkeletonVis(server, "orig_p1_ghost", (80, 155, 255))

    robot_root = server.scene.add_frame("/g1_robot", show_axes=False)
    urdf = yourdfpy.URDF.load(str(args.urdf), load_meshes=True, build_scene_graph=True)
    viser_robot = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/g1_robot")
    viser_robot.show_visual = True

    sp_partner_palm = server.scene.add_icosphere("/pts/partner_palm", radius=0.032, color=(50, 220, 90))
    sp_lstm_pred = server.scene.add_icosphere("/pts/lstm_pred", radius=0.028, color=(255, 220, 40))
    sp_blend_tgt = server.scene.add_icosphere("/pts/blend_target", radius=0.028, color=(255, 65, 65))
    sp_g1_palm = server.scene.add_icosphere("/pts/g1_palm", radius=0.030, color=(180, 80, 255))

    state = {
        "seq_idx": 0,
        "scheme": scheme_names[-1],  # default to D_Route2_协商规律
        "t": 0.0,
        "playing": True,
        "speed": 0.6,
        "show_p2": True,
        "show_p1_ghost": False,
        "show_robot": True,
        "show_pts": True,
    }

    with server.gui.add_folder("实验对比控制"):
        dd_seq = server.gui.add_dropdown("测试序列", options=seq_names, initial_value=seq_names[0])
        dd_sch = server.gui.add_dropdown("方案", options=scheme_names, initial_value=state["scheme"])

    with server.gui.add_folder("播放控制"):
        slider = server.gui.add_slider("帧", 0, max(len(records[0]["j2_full"]) - 1, 1), 1, initial_value=0)
        cb_play = server.gui.add_checkbox("自动播放", True)
        sl_speed = server.gui.add_slider("速度", 0.1, 2.0, 0.1, initial_value=0.6)

    with server.gui.add_folder("显示开关"):
        cb_p2 = server.gui.add_checkbox("真人伙伴 P2 (橙)", True)
        cb_p1 = server.gui.add_checkbox("原始真人 P1 幽灵 (蓝)", False)
        cb_rb = server.gui.add_checkbox("G1 机器人", True)
        cb_pt = server.gui.add_checkbox("手部目标点 (绿=真手/黄=LSTM/红=融合/紫=G1手)", True)

    with server.gui.add_folder("当前帧与方案读数"):
        txt_info = server.gui.add_text("方案指标", initial_value="-")
        txt_dist = server.gui.add_text("当前掌心距", initial_value="-")
        txt_pred = server.gui.add_text("LSTM预测到终点误差", initial_value="-")

    def _refresh_seq_bounds() -> None:
        r = records[state["seq_idx"]]
        max_t = max(len(r["j2_full"]) - 1, 1)
        slider.max = max_t
        if slider.value > max_t:
            slider.value = 0
            state["t"] = 0.0

    def _on_seq(_):
        state["seq_idx"] = seq_names.index(dd_seq.value)
        state["t"] = 0.0
        slider.value = 0
        _refresh_seq_bounds()

    dd_seq.on_update(_on_seq)
    dd_sch.on_update(lambda _: state.update(scheme=dd_sch.value))
    cb_play.on_update(lambda _: state.update(playing=cb_play.value))
    sl_speed.on_update(lambda _: state.update(speed=sl_speed.value))
    cb_p2.on_update(lambda _: state.update(show_p2=cb_p2.value))
    cb_p1.on_update(lambda _: state.update(show_p1_ghost=cb_p1.value))
    cb_rb.on_update(lambda _: state.update(show_robot=cb_rb.value))
    cb_pt.on_update(lambda _: state.update(show_pts=cb_pt.value))

    print(f"[viser] Route 2 Viewer running on http://0.0.0.0:{args.port} ({len(records)} test sequences)", flush=True)
    last = time.time()
    while True:
        now = time.time()
        dt = now - last
        last = now

        rec = records[state["seq_idx"]]
        n_frames = len(rec["j2_full"])
        if state["playing"]:
            state["t"] = (state["t"] + dt * args.fps * state["speed"]) % n_frames
            slider.value = int(state["t"])
        t = min(int(slider.value), n_frames - 1)

        sc = rec["schemes"][state["scheme"]]
        q_t = sc["q_traj"][min(t, len(sc["q_traj"]) - 1)]

        sk_partner.update(rec["j2_full"][t], state["show_p2"])
        sk_orig.update(rec["j1_full"][t], state["show_p1_ghost"])

        robot_root.visible = state["show_robot"]
        if state["show_robot"]:
            robot_root.position = (float(q_t[0]), float(q_t[1]), float(q_t[2]))
            robot_root.wxyz = tuple(float(v) for v in q_t[3:7])
            viser_robot.update_cfg(np.asarray(q_t[7:36], dtype=float))

        show_pts = state["show_pts"]
        p_true = rec["partner_palm"][t]
        p_lstm = rec["lstm_pred"][t]
        p_blend = rec["blend_target"][t]
        p_g1 = sc["palm_traj"][min(t, len(sc["palm_traj"]) - 1)]

        for sp, pos in (
            (sp_partner_palm, p_true),
            (sp_lstm_pred, p_lstm),
            (sp_blend_tgt, p_blend),
            (sp_g1_palm, p_g1),
        ):
            sp.visible = show_pts
            if show_pts:
                sp.position = tuple(float(v) for v in pos)

        txt_info.value = (
            f"终误差={sc['err_cm']:.1f}cm | 合法={sc['valid']} | "
            f"伸展={sc['ext']:.2f} | 倾角={sc['lean_deg']:.1f}°"
        )
        txt_dist.value = f"{np.linalg.norm(p_g1 - p_true) * 100:.1f} cm"
        txt_pred.value = f"{np.linalg.norm(p_lstm - rec['partner_palm'][-1]) * 100:.1f} cm"

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
