"""Step 3 (Layer D): Causal Partner Hand Predictor (LSTM) + Sigmoid Target Blending.

Implements Hand Reaching (2103.00616) Sec. III-B & Eq. 6:
- Input: causal history of partner's 15 upper-body 3D joints (14 symmetric upper-body
  joints + active palm, normalized relative to partner's initial pelvis origin).
- Output: predicted final 3D hand clasp location h_hat_t (relative to same origin).
- Online blending:
    h_star_t = (1 - sigma(t + alpha)) * h_hat_t + sigma(t + alpha) * h_curr_t
  smoothly transitions from early LSTM prediction to live tracked hand location.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import HISTORY_FRAMES


class PartnerHandLSTM(nn.Module):
    """Lightweight causal LSTM predicting partner's final 3D palm location."""

    def __init__(self, in_dim: int = 45, hidden_dim: int = 96, num_layers: int = 2,
                 dropout: float = 0.15) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, in_dim)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _partner_heading_rot(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin_3d, R_world_to_local_3x3) aligned to partner's t=0 facing +X."""
    origin = sample["partner_pelvis"][0].astype(np.float32)
    # partner_ub indices: 7 = SHOULDER_L, 10 = SHOULDER_R
    ub0 = sample["partner_ub"][0]
    lr = ub0[10] - ub0[7]
    z_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fwd = np.cross(lr, z_up)
    fwd[2] = 0.0
    n = float(np.linalg.norm(fwd))
    if n < 1e-5:
        yaw = 0.0
    else:
        fwd /= n
        yaw = float(np.arctan2(fwd[1], fwd[0]))
    c, s = np.cos(-yaw), np.sin(-yaw)
    r_w2l = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return origin, r_w2l


def build_sequence_windows(samples: list[dict], history: int = HISTORY_FRAMES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal sliding windows (X, Y_final, Y_curr) in partner's heading-aligned local frame."""
    xs, ys_final, ys_curr = [], [], []
    for s in samples:
        origin, r_w2l = _partner_heading_rot(s)
        ub14 = s["partner_ub"]  # (L, 14, 3)
        palm = s["partner_palm"][:, None, :]  # (L, 1, 3)
        pts_world = np.concatenate([ub14, palm], axis=1) - origin[None, None, :]
        pts_local = pts_world @ r_w2l.T
        feats = pts_local.reshape(len(pts_local), -1)  # (L, 45)
        palm_local = (s["partner_palm"] - origin[None, :]) @ r_w2l.T
        target_final = palm_local[-1]

        for t in range(len(feats)):
            # Causal left-padding with frame 0 so prediction works from t=0 onward
            start = t - history + 1
            if start < 0:
                pad = np.repeat(feats[0:1], -start, axis=0)
                win = np.concatenate([pad, feats[:t + 1]], axis=0)
            else:
                win = feats[start:t + 1]
            xs.append(win)
            ys_final.append(target_final)
            ys_curr.append(palm_local[t])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys_final, dtype=np.float32),
        np.asarray(ys_curr, dtype=np.float32),
    )


@dataclass
class TrainedPredictor:
    model: PartnerHandLSTM
    x_mean: np.ndarray
    x_std: np.ndarray
    device: str

    def predict_sequence(self, sample: dict, history: int = HISTORY_FRAMES) -> np.ndarray:
        """Predict final partner palm position in world Z-up at every causal frame t."""
        origin, r_w2l = _partner_heading_rot(sample)
        pts_world = np.concatenate([sample["partner_ub"], sample["partner_palm"][:, None, :]], axis=1) - origin[None, None, :]
        pts_local = pts_world @ r_w2l.T
        feats = pts_local.reshape(len(pts_local), -1)
        wins = []
        for t in range(len(feats)):
            start = t - history + 1
            if start < 0:
                pad = np.repeat(feats[0:1], -start, axis=0)
                win = np.concatenate([pad, feats[:t + 1]], axis=0)
            else:
                win = feats[start:t + 1]
            wins.append(win)
        x = (np.asarray(wins, dtype=np.float32) - self.x_mean) / self.x_std
        self.model.eval()
        with torch.no_grad():
            pred_local = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        return (pred_local @ r_w2l) + origin[None, :]


def blend_target(h_hat: np.ndarray, h_curr: np.ndarray, t: int, t_est: int = 30,
                 alpha_frac: float = 0.5, k_steep: float = 8.0) -> np.ndarray:
    """Sigmoid transition from LSTM predicted final location to live tracked hand (Eq. 6)."""
    z = (t / max(t_est, 1)) - alpha_frac
    sigma = 1.0 / (1.0 + np.exp(-k_steep * z))
    return (1.0 - sigma) * h_hat + sigma * h_curr


def train_lstm_predictor(
    train_samples: list[dict],
    val_samples: list[dict],
    history: int = HISTORY_FRAMES,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[TrainedPredictor, dict]:
    """Train PartnerHandLSTM with train-only normalization and val early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_tr, y_tr, _ = build_sequence_windows(train_samples, history)
    x_va, y_va, y_va_curr = build_sequence_windows(val_samples, history)

    x_mean = x_tr.mean(axis=(0, 1), keepdims=True)
    x_std = x_tr.std(axis=(0, 1), keepdims=True) + 1e-6

    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr_n), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    x_va_t = torch.from_numpy(x_va_n).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    model = PartnerHandLSTM(in_dim=x_tr.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 15
    no_imp = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(x_va_t)
            val_mse = float(loss_fn(pred_va, y_va_t).item())
            val_err_cm = float(torch.norm(pred_va - y_va_t, dim=1).mean().item() * 100.0)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = model(x_va_t).cpu().numpy()
    lstm_err_cm = float(np.linalg.norm(pred_va - y_va, axis=1).mean() * 100.0)
    naive_curr_cm = float(np.linalg.norm(y_va_curr - y_va, axis=1).mean() * 100.0)

    metrics = {
        "n_train_windows": int(len(x_tr)),
        "n_val_windows": int(len(x_va)),
        "val_lstm_err_cm": round(lstm_err_cm, 2),
        "val_no_pred_err_cm": round(naive_curr_cm, 2),
        "epochs_ran": ep + 1,
    }
    return TrainedPredictor(model=model, x_mean=x_mean, x_std=x_std, device=device), metrics
