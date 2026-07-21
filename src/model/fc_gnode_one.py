"""
model/fc_gnode.py
==================
Fully Connected GN-ODE.

This is your original FCGNOde class exactly as pasted (from
train_fc_gnode.py), unmodified. It was already resolution-agnostic —
N is a constructor argument, not hardcoded — so the same class runs
at N=24 (district) or N=274 (beat) with no code changes.
"""

import math
import torch
import torch.nn as nn


def _sigmoid_inv(x: float) -> float:
    return math.log(x / (1.0 - x))


class FCGNOde(nn.Module):
    def __init__(self, N, alpha=0.3, beta=0.6):
        super().__init__()

        # Row-normalised all-ones matrix so each unit receives
        # the average neighbour signal instead of the raw sum.
        A = torch.ones(N, N) / N
        self.register_buffer("A", A)

        # Per-unit alpha in (0,1) and beta in (0,2)
        self.raw_alpha = nn.Parameter(
            torch.full((N,), _sigmoid_inv(alpha), dtype=torch.float32)
        )
        self.raw_beta = nn.Parameter(
            torch.full((N,), _sigmoid_inv(beta / 2.0), dtype=torch.float32)
        )

        self.gnn = nn.Linear(N, N, bias=False)
        with torch.no_grad():
            self.gnn.weight.data *= 0.01

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return 2.0 * torch.sigmoid(self.raw_beta)

    def forward(self, t, C, L):
        return (
            self.alpha * C * (1 - C)   # per-unit logistic self-growth
            + self.gnn(self.A @ C)     # normalised fully connected diffusion
            - self.beta * L * C        # per-unit enforcement suppression
        )