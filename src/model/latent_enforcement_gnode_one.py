"""
model/latent_enforcement_gnode.py
==================================
GNODE with latent enforcement signal L(t) learned via small MLP.

Adjacency is symmetrically normalised, W = D^(-1/2) A D^(-1/2), matching
Eq. (8) of the paper, with self-loops excluded (A_ii = 0) so Theorem
3.8's spectral bound lambda_max(W) <= 1 applies. This replaces an
earlier version of this file that used the raw, unnormalised adjacency
directly, which did not satisfy that bound at all.

N is inferred from A.shape[0], so the same class runs at N=24 (district)
or N=274 (beat) with no code changes.

L(t,i) is predicted from time features, generalising to unseen months.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def _sigmoid_inv(x: float) -> float:
    return math.log(x / (1.0 - x))


def _symmetric_normalize(A: torch.Tensor) -> torch.Tensor:
    """W = D^(-1/2) A D^(-1/2). Assumes A has no self-loops."""
    deg = A.sum(dim=1).clamp(min=1e-8)
    d_inv_sqrt = deg.pow(-0.5)
    return A * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)


class LatentEnforcementGNODE(nn.Module):
    """
    Graph Neural ODE with latent enforcement signal.

    dC/dt = alpha[i] * C * (1-C)
            + GNN(A, C)
            - beta[i] * L(t,i) * C

    L(t,i) is output by a small MLP over time features,
    so it generalises to test months unseen during training.

    forward(t, y) interface — y is packed (N, 2):
        y[:, 0] = C   crime intensity
        y[:, 1] = L   placeholder (ignored — L comes from MLP)
    """

    def __init__(self, A, T, hidden_dim=32, alpha=0.3, beta=0.6):
        """
        Parameters
        ----------
        A          : np.ndarray  (N, N)  spatial adjacency (symmetrically
                                          normalised internally; self-loops
                                          are stripped regardless of input)
        T          : int         total number of months in dataset
        hidden_dim : int         GNN hidden size
        alpha      : float       initial self-growth rate
        beta       : float       initial suppression rate
        """
        super().__init__()

        A_tensor = torch.tensor(A, dtype=torch.float32)
        N = A_tensor.shape[0]

        # Enforce no self-loops, then symmetrically normalise, matching
        # Eq. (8) and the assumptions behind Theorem 3.8.
        A_tensor = A_tensor * (1.0 - torch.eye(N))
        A_norm = _symmetric_normalize(A_tensor)
        self.register_buffer("A", A_norm)

        self.N = N
        self.T = T

        # ── Per-unit alpha in (0, 1) ──────────────────────────────────────
        self.raw_alpha = nn.Parameter(
            torch.full((N,), _sigmoid_inv(alpha), dtype=torch.float32)
        )

        # ── Per-unit beta in (0, 2) ───────────────────────────────────────
        self.raw_beta = nn.Parameter(
            torch.full((N,), _sigmoid_inv(beta / 2.0), dtype=torch.float32)
        )

        # ── GNN layer ─────────────────────────────────────────────────────
        self.gnn = nn.Linear(N, N, bias=False)
        with torch.no_grad():
            self.gnn.weight.data *= 0.1

        # ── Unit embeddings — one per spatial unit ────────────────────────
        # captures spatial heterogeneity in enforcement
        self.beat_emb = nn.Embedding(N, 8)

        # ── Time encoder MLP ──────────────────────────────────────────────
        # input : [sin(2π t/T), cos(2π t/T), t/T, beat_emb(8)] = 11 dims
        # output: L(t,i) in (0,1) via sigmoid
        self.L_net = nn.Sequential(
            nn.Linear(11, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    @property
    def alpha(self):
        """Per-unit self-growth rate, in (0, 1). Shape: (N,)"""
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        """Per-unit enforcement suppression, in (0, 2). Shape: (N,)"""
        return 2.0 * torch.sigmoid(self.raw_beta)

    def _time_features(self, t_idx):
        """
        Encode month index t_idx into continuous time features.
        Returns shape (N, 11) — one feature vector per unit.
        """
        t_norm = t_idx / self.T
        sin_t  = torch.tensor([np.sin(2 * np.pi * t_norm)],
                               dtype=torch.float32,
                               device=self.beat_emb.weight.device)
        cos_t  = torch.tensor([np.cos(2 * np.pi * t_norm)],
                               dtype=torch.float32,
                               device=self.beat_emb.weight.device)
        t_tens = torch.tensor([t_norm],
                               dtype=torch.float32,
                               device=self.beat_emb.weight.device)

        # time features broadcast across all N units
        time_feat = torch.stack([sin_t, cos_t, t_tens], dim=1)  # (1, 3)
        time_feat = time_feat.expand(self.N, -1)                 # (N, 3)

        # unit embeddings
        beat_idx  = torch.arange(self.N, device=self.beat_emb.weight.device)
        beat_feat = self.beat_emb(beat_idx)                      # (N, 8)

        return torch.cat([time_feat, beat_feat], dim=1)          # (N, 11)

    @property
    def L(self):
        """
        Full L matrix shape (T, N) — for analysis and logging.
        Computed over all months using the MLP.
        """
        rows = []
        for t in range(self.T):
            feat = self._time_features(t)
            L_t  = self.L_net(feat).squeeze(-1)   # (N,)
            rows.append(L_t)
        return torch.stack(rows, dim=0)            # (T, N)

    def forward(self, t, y, t_idx):
        """
        ODE right-hand side.

        Parameters
        ----------
        t     : torch.Tensor   scalar (unused, required by torchdiffeq)
        y     : torch.Tensor   (N, 2)
                               y[:, 0] = C  crime intensity
                               y[:, 1] = placeholder (L comes from MLP)
        t_idx : int            month index for time feature encoding

        Returns
        -------
        dy : torch.Tensor  (N, 2)
        """
        C = y[:, 0]   # (N,)

        feat = self._time_features(t_idx)          # (N, 11)
        L_t  = self.L_net(feat).squeeze(-1)        # (N,)

        dC = (
            self.alpha * C * (1 - C)
            + self.gnn(self.A @ C)
            - self.beta * L_t * C
        )

        # second channel is placeholder — return zeros
        dL = torch.zeros_like(C)

        return torch.stack([dC, dL], dim=1)        # (N, 2)

    def param_summary(self):
        a = self.alpha.detach()
        b = self.beta.detach()
        print(f"alpha — min: {a.min():.4f}  max: {a.max():.4f}  mean: {a.mean():.4f}")
        print(f"beta  — min: {b.min():.4f}  max: {b.max():.4f}  mean: {b.mean():.4f}")

    def get_learned_enforcement(self, units, months, id_col="unit"):
        L_np = self.L.detach().cpu().numpy()   # (T, N)
        rows = []
        for t, m in enumerate(months):
            for i, u in enumerate(units):
                rows.append({
                    "month"   : m,
                    id_col    : u,
                    "L_latent": float(L_np[t, i]),
                })
        return pd.DataFrame(rows)