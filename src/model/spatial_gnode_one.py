"""
model/spatial_gnode.py
=======================
Spatial Graph GN-ODE.

Adjacency is symmetrically normalised, W = D^(-1/2) A D^(-1/2), matching
Eq. (8) of the paper, with self-loops explicitly excluded (A_ii = 0) so
that Theorem 3.8's spectral bound lambda_max(W) <= 1 applies as stated.
This replaces the row-normalization used in an earlier version of this
file, which did not satisfy that bound in general.

N is inferred from A.shape[0], so the same class runs at N=24 (district)
or N=274 (beat) with no code changes.
"""

import math
import torch
from torch import nn


def _sigmoid_inv(x: float) -> float:
    """Inverse sigmoid — maps a value in (0, 1) to its raw logit."""
    return math.log(x / (1.0 - x))


def _symmetric_normalize(A: torch.Tensor) -> torch.Tensor:
    """W = D^(-1/2) A D^(-1/2). Assumes A has no self-loops."""
    deg = A.sum(dim=1).clamp(min=1e-8)
    d_inv_sqrt = deg.pow(-0.5)
    return A * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)


class GNODE(nn.Module):
    def __init__(self, A, alpha=0.3, beta=0.6, hidden=16):
        super().__init__()

        A_tensor = torch.tensor(A, dtype=torch.float32)
        N = A_tensor.shape[0]

        # Enforce no self-loops regardless of what the input A contains,
        # so the Gershgorin/Perron-Frobenius bound in Theorem 3.8 holds.
        A_tensor = A_tensor * (1.0 - torch.eye(N))
        A_norm = _symmetric_normalize(A_tensor)
        self.register_buffer("A", A_norm)
        self.N = N

        # ── Per-unit alpha in (0, 1) ──────────────────────────────────────
        self.raw_alpha = nn.Parameter(
            torch.full((N,), _sigmoid_inv(alpha), dtype=torch.float32)
        )

        # ── Per-unit beta in (0, 2) ───────────────────────────────────────
        self.raw_beta = nn.Parameter(
            torch.full((N,), _sigmoid_inv(beta / 2.0), dtype=torch.float32)
        )

        # ── Node-wise GNN transform after spatial aggregation ─────────────
        self.gnn = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

        # Small init — keeps early ODE dynamics stable
        for layer in self.gnn:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return 2.0 * torch.sigmoid(self.raw_beta)

    def forward(self, t, y):
        """
        ODE right-hand side.

        Parameters
        ----------
        t : torch.Tensor   scalar (unused, required by torchdiffeq)
        y : torch.Tensor   (N, 2)
                           y[:, 0] = C  crime intensity
                           y[:, 1] = L  enforcement level  (held fixed per step)

        Returns
        -------
        dy : torch.Tensor  (N, 2)
        """
        C = y[:, 0]
        L = y[:, 1]

        # Spatial aggregation — A is symmetrically normalised
        neighbor_agg = (self.A @ C).unsqueeze(-1)           # (N, 1)
        spatial      = self.gnn(neighbor_agg).squeeze(-1)   # (N,)

        dC = (
            self.alpha * C * (1 - C)
            + spatial
            - self.beta * L * C
        )

        # dL is zero — L is a fixed external input per time-step
        dL = torch.zeros_like(L)

        return torch.stack([dC, dL], dim=1)                 # (N, 2)

    def param_summary(self):
        a = self.alpha.detach()
        b = self.beta.detach()
        print(f"alpha — min: {a.min():.4f}  max: {a.max():.4f}  mean: {a.mean():.4f}")
        print(f"beta  — min: {b.min():.4f}  max: {b.max():.4f}  mean: {b.mean():.4f}")