"""
model/fc_gnode.py
==================
Fully Connected GN-ODE.

Architectural fix (see conversation history / manuscript Section 6,
"A note on Fully Connected GNODE's training dynamics"): the original
version used gnn = nn.Linear(N, N, bias=False) applied to A @ C.
Because A = ones(N,N)/N has identical rows, A @ C collapses to a
vector where every entry equals mean(C) -- a single scalar repeated
N times. Feeding that into an N x N dense layer means N^2 parameters
are used to represent what is mathematically only an N-parameter
function (each unit's own response to one shared scalar input).
That redundant parameterisation (e.g. 576 params at N=24 for what
needs only 24) is the most likely cause of the immediate
overfitting observed in training (validation loss best at epoch 1,
monotonically worse afterward).

Fix: since the true input is a single scalar (mean(C)), gnn is now
Linear(1, N, bias=False) -- mathematically equivalent expressive
capacity for this architecture, with N parameters instead of N^2.
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
        # NOTE: every row is identical (1/N), so A @ C always
        # collapses to a scalar (mean(C)) repeated N times.
        A = torch.ones(N, N) / N
        self.register_buffer("A", A)
        self.N = N

        # Per-unit alpha in (0,1) and beta in (0,2)
        self.raw_alpha = nn.Parameter(
            torch.full((N,), _sigmoid_inv(alpha), dtype=torch.float32)
        )
        self.raw_beta = nn.Parameter(
            torch.full((N,), _sigmoid_inv(beta / 2.0), dtype=torch.float32)
        )

        # Right-sized parameterisation: N params instead of N^2,
        # since the true input dimensionality is 1 (mean(C)), not N.
        self.gnn = nn.Linear(1, N, bias=False)
        with torch.no_grad():
            self.gnn.weight.data *= 0.01

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return 2.0 * torch.sigmoid(self.raw_beta)

    def forward(self, t, C, L):
        # A @ C is a length-N vector with every entry equal to
        # mean(C); extract that scalar directly rather than
        # re-deriving it via a full N x N matmul + dense layer.
        mean_C = (self.A @ C)[0].view(1)          # scalar, shape (1,)
        spatial = self.gnn(mean_C)                # shape (N,)

        return (
            self.alpha * C * (1 - C)   # per-unit logistic self-growth
            + spatial                  # per-unit response to mean crime level
            - self.beta * L * C        # per-unit enforcement suppression
        )