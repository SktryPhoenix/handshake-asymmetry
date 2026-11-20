# Route 2 Codebase: ProMP + LSTM Partner Prediction + Embodiment-Aware Search

将以下两篇轻量级路线论文与我们在 `research/law/` 量出的人类握手协商规律结合，直接落地到 `Inter-X A001 + Unitree G1`：
- **Paper A**：*Hand Reaching for Handshaking* (`2103.00616`) —— LSTM 预测伙伴落点 + Sigmoid 目标平滑过渡 + ProMP 概率运动基元
- **Paper B**：*Embodiment-Aware Inference-Time Search* (`2610.15274`) —— 采样 $K$ 个候选反应、经 G1 重定向后用可行性与交互保真度筛选最优解

## 三层归属映射

| 模块 | 所属层 | 作用 |
|---|---|---|
| `data_prep.py` | 数据基建 | 用 `raw_betas` 恢复真实体型、每人独立对地、按 `G` 组防泄漏切分 |
| `promp.py` | **生成 / 先验层 (P)** | 从人类 4-DoF 手臂示范拟合 $\mathcal{N}(\mu_\omega, \Sigma_\omega)$，提供平滑先验与 $K$ 候选采样 |
| `predictor_lstm.py` | **部署层 (D)** | 因果滑窗预测对方手掌最终落点 $\hat{h}_t$，并用 Sigmoid 融合实时跟踪 $h_t$ |
| `g1_kinematics.py` | **重定向 / 控制层 (R/D)** | 基于 `g1_29dof.xml` 的接地、25° 腰部前倾、DLS 手臂 IK 及 MuJoCo 碰撞/限位校验 $r_{t,k}$ |
| `reaction_search.py` | **重定向层 (R)** | $K=16$ 候选生成、合法集过滤 $E_t$、双空间打分 $(B_{\text{kin}}+B_{\text{int}})/2$ 与让步预算计算 |
| `run_pipeline.py` | 端到端实验 | 一键跑通 4 组消融实验（A 朴素 / B Paper A / C Paper A+B / D 完整协商路线） |

## 3090 快速上手

```bash
ssh 3090
source ~/repos/OmniRetarget/scripts/source_retargeting_setup.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# 1) 同步本地 codebase 到服务器
# （在本地执行）tar czf - -C ~/Documents/deepseek-harness/default-workspace/research v3_route2_promp_search | ssh 3090 'tar xzf - -C ~/01-omniretarget-first-attempt/'

# 2) 先用 30 对快速烟雾测试（约 1 分钟）
cd ~/01-omniretarget-first-attempt/v3_route2_promp_search
python3 run_pipeline.py --max-pairs 30 --epochs 30 --work-dir ~/runs/route2_smoke

# 3) 全量 324 对（178 对有效接触）完整跑通
python3 run_pipeline.py --epochs 80 --k-search 16 --work-dir ~/runs/handshake_route2
```
