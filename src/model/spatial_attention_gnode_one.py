"""
model/spatial_attention_gnode.py
=================================
Spatial Attention GN-ODE — ONE file, ONE class, used at both district
(N=24) and beat (N=274) resolution.

This is NOT a new architecture. It is your two previously-separate,
individually-proven files (the district model and BeatAttentionGNODEv2)
packaged behind one class and one forward(t, C, L) interface, auto-selected
by N at construction time. The attention math in each branch is unchanged
from the original files — this is a refactor of the calling convention,
not a redesign of the dynamics.

Parameter convention: alpha_i, beta_i are learned per spatial unit at
BOTH resolutions, matching the Fully Connected, Spatial Graph, and Latent
Enforcement variants (Assumption 2.1). This differs from an earlier
version of this file, which used a single shared scalar (alpha, beta)
at district scale only; that special case has been removed so all four
GNODE variants share the same per-unit parameterisation uniformly.

Why one architecture doesn't naturally work at both scales (see
conversation history for the full empirical trail): a small model
built around a single scalar C_i (the district design) works at N=24
because there's only a handful of neighbours to discriminate between,
but at N=274 many beats end up with similar normalised crime
intensities, so attention scores collapse toward uniform. A rich model
with a learned per-unit embedding (the beat design) avoids that
collapse, but its one N-scaling parameter (the embedding table) gets
far less gradient signal per epoch at N=24 than at N=274, causing
early stopping at a poor optimum. Four attempts at a single merged
architecture (degree features, learned gamma, scalar-only, per-unit
vectors) each failed to reproduce the district model's proven quality,
which is why this version does not invent new math — it reuses exactly
what already works at each scale, apart from the alpha/beta
parameterisation change noted above.

Interface: forward(t, C, L) is now uniform at both resolutions. The
beat branch's month index, previously a third positional argument,
is instead read from a settable attribute — call set_time_idx(i)
once before each odeint solve. At district scale this call is a
harmless no-op.
"""

import math
import numpy as np
import torch
import torch.nn as nn


def _sigmoid_inv(x: float) -> float:
    return math.log(x / (1.0 - x))


class SpatialAttentionGNODE(nn.Module):
    # Districts: N=24 (small-N branch). Beats: N=274 (large-N branch).
    # Any resolution with N above this uses the beat-style architecture;
    # at or below it, the district-style architecture is used.
    LARGE_N_THRESHOLD = 50

    def __init__(self, A, hidden_dim=32, T=None):
        super().__init__()

        A_tensor = torch.tensor(A, dtype=torch.float32)
        N = A_tensor.shape[0]
        self.N = N
        self.large_n = N > self.LARGE_N_THRESHOLD

        A_self = A_tensor + torch.eye(N)
        self.register_buffer("A", A_self)
        self.register_buffer("mask", A_self > 0)
        self.register_buffer("scale", torch.sqrt(torch.tensor(float(hidden_dim))))

        if not self.large_n:
            # ── Small-N branch: per-unit alpha/beta, matching the other
            # three GNODE variants (Assumption 2.1). gamma remains a
            # single shared scalar gate on the attention output — it is
            # not a growth/enforcement rate, so it is unaffected by this
            # change and is not covered by Assumption 2.1.
            self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(0.3), dtype=torch.float32))
            self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(0.3), dtype=torch.float32))
            self.raw_gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

            self.Wq  = nn.Linear(1, hidden_dim)
            self.Wk  = nn.Linear(1, hidden_dim)
            self.Wv  = nn.Linear(1, hidden_dim)
            self.out = nn.Linear(hidden_dim, 1)

        else:
            # ── Large-N branch: your original proven BeatAttentionGNODEv2 ──
            assert T is not None, "T (total months) is required when N > LARGE_N_THRESHOLD"
            self.T = T
            self._t_idx = 0   # set via set_time_idx() before each odeint call

            self.raw_alpha = nn.Parameter(torch.full((N,), _sigmoid_inv(0.3), dtype=torch.float32))
            self.raw_beta  = nn.Parameter(torch.full((N,), _sigmoid_inv(0.3), dtype=torch.float32))

            self.beat_emb = nn.Embedding(N, 8)

            # input: [C(1), L(1), beat_emb(8), sin(1), cos(1), t_norm(1)] = 13
            self.Wq  = nn.Linear(13, hidden_dim)
            self.Wk  = nn.Linear(13, hidden_dim)
            self.Wv  = nn.Linear(13, hidden_dim)
            self.out = nn.Linear(hidden_dim, 1)

            for layer in (self.Wq, self.Wk, self.Wv, self.out):
                layer.weight.data *= 0.1
                layer.bias.data.zero_()

    def set_time_idx(self, t_idx: int):
        """Call once before each odeint solve. No-op at district scale."""
        if self.large_n:
            self._t_idx = t_idx

    @property
    def alpha(self):
        """Per-unit self-growth rate, in (0, 1). Shape: (N,) at both resolutions."""
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        """Per-unit enforcement suppression, in (0, 1). Shape: (N,) at both resolutions."""
        return torch.sigmoid(self.raw_beta)

    @property
    def gamma(self):
        """Shared scalar gate on the attention spatial term; district scale only."""
        if not self.large_n:
            return torch.sigmoid(self.raw_gamma)
        return None

    def forward(self, t, C, L):
        if not self.large_n:
            # ── District forward pass — alpha, beta now (N,) instead of
            # scalar; elementwise ops below broadcast unchanged. ──
            N = C.shape[0]
            C_in = C.view(N, 1)

            Q = self.Wq(C_in)
            K = self.Wk(C_in)
            V = self.Wv(C_in)

            scores = torch.matmul(Q, K.T) / self.scale
            scores = scores.masked_fill(~self.mask, -1e9)
            attn   = torch.softmax(scores, dim=1)

            spatial_term = self.out(torch.matmul(attn, V)).view(-1)

            return (
                self.alpha * C * (1 - C)
                + self.gamma * spatial_term
                - self.beta * L * C
            )

        else:
            # ── Exact original beat forward pass, adapted only so that
            # t_idx comes from self._t_idx instead of a 3rd positional
            # argument, and L is taken directly rather than unpacked
            # from a packed (N,2) state (mathematically identical). ──
            t_norm = self._t_idx / self.T
            device = C.device

            sin_t  = torch.full((self.N, 1), float(np.sin(2 * np.pi * t_norm)),
                                 dtype=torch.float32, device=device)
            cos_t  = torch.full((self.N, 1), float(np.cos(2 * np.pi * t_norm)),
                                 dtype=torch.float32, device=device)
            t_feat = torch.full((self.N, 1), float(t_norm),
                                 dtype=torch.float32, device=device)

            beat_idx  = torch.arange(self.N, device=device)
            beat_feat = self.beat_emb(beat_idx)   # (N, 8)

            feat = torch.cat([
                C.unsqueeze(-1), L.unsqueeze(-1), beat_feat, sin_t, cos_t, t_feat,
            ], dim=1)   # (N, 13)

            Q = self.Wq(feat)
            K = self.Wk(feat)
            V = self.Wv(feat)

            scores = torch.matmul(Q, K.T) / self.scale
            scores = scores.masked_fill(~self.mask, -1e9)
            attn   = torch.softmax(scores, dim=1)

            spatial = self.out(torch.matmul(attn, V)).squeeze(-1)

            return (
                self.alpha * C * (1 - C)
                + spatial
                - self.beta * L * C
            )

    def param_summary(self):
        a, b = self.alpha.detach(), self.beta.detach()
        print(f"alpha — min: {a.min():.4f}  max: {a.max():.4f}  mean: {a.mean():.4f}")
        print(f"beta  — min: {b.min():.4f}  max: {b.max():.4f}  mean: {b.mean():.4f}")
        if not self.large_n:
            print(f"gamma = {self.gamma.item():.4f}")