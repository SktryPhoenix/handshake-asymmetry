# Core Implementation (`src/`): ProMP + Causal LSTM + Embodiment-Aware Search

将以下两篇论文与我们在 `analysis/` 量出的人类握手几何协商规律结合，落地到 `Inter-X A001 + Unitree G1`：
- **Paper A**：*Hand Reaching for Handshaking* (`arXiv:2103.00616`) —— 因果 LSTM 预测伙伴手掌落点 + Sigmoid 目标平滑融合 + ProMP 概率运动基元
- **Paper B**：*Embodiment-Aware Inference-Time Search* (`arXiv:2610.15274`) —— 采样 $K=16$ 个候选反应、经 G1 重定向与碰撞校验后用双空间打分筛选最优解
- **Negotiation Law (Ours)**：注入 $25^\circ$ 躯干前倾先验、$0.84\text{ m}$ 站距与 $[0.84, 0.90]$ 手臂微曲舒适伸展带约束

## 模块与三层归属映射

| 模块 | 所属层 | 作用 |
|---|---|---|
| `data_prep.py` | 数据基建 | 用 `raw_betas` 恢复真实体型、每人独立对地、同侧握手配对（`R-R`/`L-L`）、按 `G` 组防泄漏切分 |
| `promp.py` | **生成 / 先验层 (P)** | 从人类 4-DoF 手臂示范拟合 $\mathcal{N}(\mu_\omega, \Sigma_\omega)$，提供平滑先验、贝叶斯条件更新与 $K$ 候选采样 |
| `predictor_lstm.py` | **部署层 (D)** | 因果滑窗预测对方手掌最终落点 $\hat{h}_t$，并用 Sigmoid 融合实时跟踪 $h_t$ |
| `g1_kinematics.py` | **重定向 / 控制层 (R/D)** | 基于 `g1_29dof.xml` 的接地、平滑腰部前倾、零空间 DLS IK 及解剖学/碰撞校验 $r_{t,k}$ |
| `reaction_search.py` | **重定向层 (R)** | $K=16$ 候选生成、合法集过滤 $E_t$、双空间打分 $(B_{\text{kin}}+B_{\text{int}})/2$ 与舒适伸展带约束 |
| `run_pipeline.py` | 端到端实验 | 一键跑通 4 组消融实验（A 朴素延迟 / B Paper A / C Paper A+B / D 完整协商路线） |
| `view_viser.py` | 交互可视化 | 启动 3D Viser Web 界面实时对比四组方案与四个目标球 |
| `record_demos.py` | Demo 导出 | 导出四面板对比 `.gif` / `.mp4` / `_contact_frame.png` 与可交互 `.viser` 3D 录制文件 |

## 运行示例

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# 1) 全量跑通四组消融实验
python3 src/run_pipeline.py --epochs 80 --k-search 16 --work-dir runs/handshake_route2

# 2) 启动 Viser 交互式 3D 预览
python3 src/view_viser.py --vis runs/handshake_route2/route2_vis.npy --port 8091

# 3) 导出 4 组对比动图、视频与 .viser 3D 录制
python3 src/record_demos.py --vis runs/handshake_route2/route2_vis.npy --out-dir demos
```
