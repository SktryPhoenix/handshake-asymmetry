"""Step 2 (Layer P): Probabilistic Movement Primitives (ProMP) for Handshake Reaching.

Implements Hand Reaching (2103.00616) Sec. II-B & III-C:
- Time-normalized RBF basis functions over phase z in [0, 1].
- Minimum-jerk ridge regression for per-demonstration weights:
    w_i = (Psi^T Psi + lam * Gamma^T Gamma)^-1 Psi^T tau_i
- Gaussian prior w ~ N(mu_w, cov_w) over 4-DoF arm trajectories (or 3D task space).
- Bayesian conditioning (Eq. 3-5) and K-candidate sampling for Reaction Search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rbf_basis_and_jerk(z: np.ndarray, n_basis: int = 8, width: float = 0.14) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized RBF basis matrix Phi (T, B) and its 3rd derivative (T, B)."""
    centers = np.linspace(-0.05, 1.05, n_basis)
    diff = z[:, None] - centers[None, :]
    inv_2w2 = 1.0 / (2.0 * width * width)
    raw = np.exp(-(diff * diff) * inv_2w2)
    denom = raw.sum(axis=1, keepdims=True) + 1e-12
    phi = raw / denom

    # Numerical 3rd derivative wrt phase z for minimum-jerk regularization
    dz = 1.0 / max(len(z) - 1, 1)
    if len(z) >= 4:
        d3 = np.gradient(np.gradient(np.gradient(phi, dz, axis=0), dz, axis=0), dz, axis=0)
    else:
        d3 = np.zeros_like(phi)
    return phi, d3


def block_diag_basis(phi: np.ndarray, n_dims: int) -> np.ndarray:
    """Build block-diagonal basis matrix Psi of shape (T * n_dims, n_dims * B)."""
    t_len, n_b = phi.shape
    psi = np.zeros((t_len * n_dims, n_dims * n_b), dtype=np.float64)
    for d in range(n_dims):
        psi[d * t_len:(d + 1) * t_len, d * n_b:(d + 1) * n_b] = phi
    return psi


def eval_basis_at(z_scalar: float, n_dims: int, n_basis: int = 8, width: float = 0.14) -> np.ndarray:
    """Return block-diagonal basis Psi(z_scalar)^T of shape (n_dims, n_dims * B)."""
    centers = np.linspace(-0.05, 1.05, n_basis)
    diff = z_scalar - centers
    raw = np.exp(-(diff * diff) / (2.0 * width * width))
    phi_z = raw / (raw.sum() + 1e-12)
    psi_t = np.zeros((n_dims, n_dims * n_basis), dtype=np.float64)
    for d in range(n_dims):
        psi_t[d, d * n_basis:(d + 1) * n_basis] = phi_z
    return psi_t


@dataclass
class ProMP:
    n_dims: int = 4
    n_basis: int = 8
    width: float = 0.14
    reg_lambda: float = 1e-6
    mu_w: np.ndarray | None = None
    cov_w: np.ndarray | None = None

    def fit(self, trajectories: list[np.ndarray]) -> "ProMP":
        """Fit N(mu_w, cov_w) over a list of (T_i, D) demonstrations."""
        weights = []
        for traj in trajectories:
            t_len, d = traj.shape
            assert d == self.n_dims, f"Expected {self.n_dims} dims, got {d}"
            z = np.linspace(0.0, 1.0, t_len)
            phi, d3 = rbf_basis_and_jerk(z, self.n_basis, self.width)
            psi = block_diag_basis(phi, self.n_dims)
            gamma = block_diag_basis(d3, self.n_dims)
            # Flatten column-major per dimension to align with block_diag_basis
            tau = traj.T.reshape(-1).astype(np.float64)
            lhs = psi.T @ psi + self.reg_lambda * (gamma.T @ gamma) + 1e-8 * np.eye(psi.shape[1])
            w_i = np.linalg.solve(lhs, psi.T @ tau)
            weights.append(w_i)

        w_mat = np.stack(weights, axis=0)
        self.mu_w = w_mat.mean(axis=0)
        self.cov_w = np.cov(w_mat, rowvar=False) + 1e-5 * np.eye(w_mat.shape[1])
        return self

    def decode(self, w: np.ndarray, t_len: int) -> np.ndarray:
        """Decode weight vector w (D*B,) into a trajectory of shape (t_len, D)."""
        z = np.linspace(0.0, 1.0, t_len)
        phi, _ = rbf_basis_and_jerk(z, self.n_basis, self.width)
        w_blocks = w.reshape(self.n_dims, self.n_basis)  # (D, B)
        return (phi @ w_blocks.T).astype(np.float32)

    def rollout_mean(self, t_len: int) -> np.ndarray:
        assert self.mu_w is not None
        return self.decode(self.mu_w, t_len)

    def sample(self, k: int, t_len: int, rng: np.random.Generator | None = None,
               include_mean: bool = True, scale: float = 1.0) -> list[np.ndarray]:
        """Sample K candidate trajectories from N(mu_w, scale^2 * cov_w)."""
        assert self.mu_w is not None and self.cov_w is not None
        rng = rng or np.random.default_rng(0)
        out: list[np.ndarray] = []
        if include_mean:
            out.append(self.decode(self.mu_w, t_len))
        need = k - len(out)
        if need > 0:
            ws = rng.multivariate_normal(self.mu_w, (scale ** 2) * self.cov_w, size=need)
            for w in ws:
                out.append(self.decode(w, t_len))
        return out

    def condition(self, z_t: float, y_target: np.ndarray,
                  obs_std: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
        """Bayesian task/joint conditioning (Eq. 3-5 of 2103.00616).

        Returns posterior (mu_w_star, cov_w_star) conditioned on observation y_target at phase z_t.
        """
        assert self.mu_w is not None and self.cov_w is not None
        psi_t = eval_basis_at(float(np.clip(z_t, 0.0, 1.0)), self.n_dims, self.n_basis, self.width)  # (D, D*B)
        sig_y = (obs_std ** 2) * np.eye(self.n_dims)
        innov_cov = sig_y + psi_t @ self.cov_w @ psi_t.T
        k_gain = self.cov_w @ psi_t.T @ np.linalg.inv(innov_cov)
        mu_star = self.mu_w + k_gain @ (y_target.astype(np.float64) - psi_t @ self.mu_w)
        cov_star = self.cov_w - k_gain @ psi_t @ self.cov_w
        return mu_star, cov_star

    def save(self, path: str) -> None:
        np.savez(path, n_dims=self.n_dims, n_basis=self.n_basis, width=self.width,
                 reg_lambda=self.reg_lambda, mu_w=self.mu_w, cov_w=self.cov_w)

    @classmethod
    def load(cls, path: str) -> "ProMP":
        d = np.load(path)
        return cls(
            n_dims=int(d["n_dims"]),
            n_basis=int(d["n_basis"]),
            width=float(d["width"]),
            reg_lambda=float(d["reg_lambda"]),
            mu_w=d["mu_w"],
            cov_w=d["cov_w"],
        )
