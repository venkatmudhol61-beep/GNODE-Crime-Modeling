import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────
def _logit(p):
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _inv_softplus(y):
    return math.log(math.expm1(y))


def _binary_graph(A):
    """Symmetric, binary, zero-diagonal adjacency."""
    if isinstance(A, torch.Tensor):
        A = A.clone().detach().float()
    else:
        A = torch.tensor(A, dtype=torch.float32)
    B = (A > 0).float()
    B = torch.maximum(B, B.T)
    B.fill_diagonal_(0.0)
    return B


def _bounded_head(d_in, hidden, p0):
    """MLP whose sigmoid output starts near p0. Input: unit embedding only."""
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    with torch.no_grad():
        net[0].weight.mul_(0.5)
        net[2].weight.mul_(0.1)
        net[2].bias.fill_(_logit(p0))
    return net


def _season_feats(moy, device):
    """Calendar-month features, period = 12 months. moy: ints in 0..11."""
    m = torch.as_tensor(moy, device=device).float().view(-1, 1)
    ang = 2.0 * math.pi * m / 12.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)       # (M, 2)


# ── base model ───────────────────────────────────────────────────────
class GNODEBase(nn.Module):
    """
    dC_i/dt = alpha_i(t) C_i (1 - C_i)
              - sum_j K_ij (C_i - C_j)
              - beta_i L_i(t) C_i

    alpha_i(t) = alpha_i * (1 + a * tanh(g_i . [sin, cos](month of year)))
                 in (alpha_i(1-a), alpha_i(1+a)), a = season_amp < 1  -> positive, bounded.

    Initial state = learned convex combination of the last `hist` observed
    months (causal smoother). hist=1 gives plain C(t_k).

    Still guaranteed by construction (no network ever sees C):
      0 < alpha_i(t) <= (1+a) alpha_max,   0 < beta_i <= beta_max
      K = K^T >= 0, zero diagonal,         L_i(t) in [0, 1]
    """

    def __init__(self, A, T, d_e=16, hidden=16,
                 alpha_max=1.0, beta_max=5.0,
                 alpha0=0.3, beta0=0.6,
                 alpha=None, beta=None,
                 hist=4, seasonal=True, season_amp=0.5):
        super().__init__()
        if alpha is not None:
            alpha0 = alpha
        if beta is not None:
            beta0 = beta

        B = _binary_graph(A)
        self.register_buffer("A", B)
        self.register_buffer("idx", torch.arange(B.shape[0]))
        self.N = B.shape[0]
        self.T = max(int(T), 1)
        self.alpha_max = float(alpha_max)
        self.beta_max = float(beta_max)

        self.emb = nn.Embedding(self.N, d_e)
        self.alpha_head = _bounded_head(d_e, hidden, alpha0 / self.alpha_max)
        self.beta_head = _bounded_head(d_e, hidden, beta0 / self.beta_max)

        # seasonal modulation of the growth rate
        self.seasonal = bool(seasonal)
        self.season_amp = float(season_amp)
        self.season_head = nn.Linear(d_e, 2)
        with torch.no_grad():
            self.season_head.weight.mul_(0.1)
            self.season_head.bias.zero_()

        # learned causal smoother for the initial state (oldest -> newest)
        self.hist = max(int(hist), 1)
        w0 = torch.tensor([0.5 ** (self.hist - 1 - j) for j in range(self.hist)])
        self.hist_logits = nn.Parameter(torch.log(w0 / w0.sum()))

        # calendar (set from the real month labels via set_calendar)
        self.register_buffer("moy_all", torch.arange(self.T) % 12, persistent=False)
        self.n_train = self.T

        self._L = None
        self._t_idx = 0
        self._moy = 0

    # ---- calendar / context -----------------------------------------
    def set_calendar(self, moy, n_train=None):
        m = torch.as_tensor(list(moy), dtype=torch.long)
        assert m.numel() == self.T, "calendar length must equal T"
        self.moy_all.copy_(m)
        if n_train is not None:
            self.n_train = int(n_train)

    def set_context(self, L, t_idx=0):
        self._L = L
        self._t_idx = int(t_idx)
        self._moy = int(self.moy_all[self._t_idx])

    def init_state(self, Cwin):
        """Cwin: (hist, N) last observed months, oldest first -> (N,)."""
        w = torch.softmax(self.hist_logits, dim=0)
        return (w[:, None] * Cwin).sum(dim=0)

    # ---- coefficients ------------------------------------------------
    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(
            self.alpha_head(self.emb(self.idx))).squeeze(-1)

    @property
    def beta(self):
        return self.beta_max * torch.sigmoid(
            self.beta_head(self.emb(self.idx))).squeeze(-1)

    def alpha_t(self):
        a = self.alpha
        if not self.seasonal:
            return a
        u = _season_feats([self._moy], a.device)[0]                  # (2,)
        g = self.season_head(self.emb(self.idx))                     # (N, 2)
        return a * (1.0 + self.season_amp * torch.tanh(g @ u))

    def coupling(self):
        raise NotImplementedError

    def enforcement(self):
        assert self._L is not None, "Call set_context(L, t_idx) before odeint"
        return self._L.view(-1).clamp(0.0, 1.0)

    # ---- vector field --------------------------------------------------
    def forward(self, t, C):
        C = C.view(-1)
        K = self.coupling()
        diffusion = K.sum(dim=1) * C - K @ C
        return (self.alpha_t() * C * (1.0 - C)
                - diffusion
                - self.beta * self.enforcement() * C)

    # ---- diagnostics ---------------------------------------------------
    @torch.no_grad()
    def _laplacian_eigs(self):
        K = (self.coupling() > 0).float()
        Lap = torch.diag(K.sum(1)) - K
        return torch.linalg.eigvalsh(Lap)

    @torch.no_grad()
    def is_connected(self):
        return bool(self._laplacian_eigs()[1] > 1e-8)

    @torch.no_grad()
    def n_components(self):
        return int((self._laplacian_eigs() < 1e-8).sum().item())

    @torch.no_grad()
    def audit(self, L_all=None):
        if L_all is None and hasattr(self, "L_all"):
            L_all = self.L_all
        a, b, K = self.alpha, self.beta, self.coupling()
        a_eff = a.max().item() * ((1.0 + self.season_amp) if self.seasonal else 1.0)
        out = {"alpha_eff_max": round(a_eff, 4),
               "beta_min": round(b.min().item(), 4),
               "kappa_max": round(K.max().item(), 4),
               "connected": self.is_connected(),
               "components": self.n_components(),
               "smooth_w": [round(x, 3) for x in
                            torch.softmax(self.hist_logits, 0).tolist()]}
        if L_all is not None:
            L_min = torch.as_tensor(L_all).clamp(0, 1).min().item()
            out["L_min"] = round(L_min, 4)
            out["gamma"] = round(out["beta_min"] * L_min - out["alpha_eff_max"], 4)
        return out

    def param_summary(self, L_all=None):
        print(f"[{self.__class__.__name__}]", self.audit(L_all))


# ── 1. Fully connected ───────────────────────────────────────────────
class GNODEfc(GNODEBase):
    def __init__(self, A, T, kappa0=0.1, **kw):
        super().__init__(A, T, **kw)
        self.raw_kappa = nn.Parameter(torch.tensor(_inv_softplus(kappa0)))

    def coupling(self):
        N, dev = self.N, self.A.device
        K = (torch.ones(N, N, device=dev) - torch.eye(N, device=dev)) / N
        return F.softplus(self.raw_kappa) * K


# ── 2. Spatial graph ─────────────────────────────────────────────────
class GNODEspatial(GNODEBase):
    def __init__(self, A, T, kappa0=0.1, **kw):
        super().__init__(A, T, **kw)
        self.raw_kappa = nn.Parameter(torch.tensor(_inv_softplus(kappa0)))

    def coupling(self):
        return F.softplus(self.raw_kappa) * self.A


# ── 3. Spatial attention (embedding-based, state-independent) ────────
class GNODEattn(GNODEBase):
    """Bounded edge re-weighting, nested in the Spatial variant.

        w_ij = kappa * A_ij * m_ij,   m_ij = 2 * sigmoid(0.5 (s_ij + s_ji)) in (0, 2)
        s_ij = (Wq e'_i)(Wk e'_j) / sqrt(d_a)

    e' are SEPARATE attention embeddings (not shared with alpha/beta/season heads),
    and Wq, Wk start small, so m_ij ~ 1 at initialisation: the model starts equal
    to the Spatial variant and can only move each edge within (0, 2) x kappa.
    Symmetric, >= 0, zero diagonal, row sums <= 2 kappa d_i; state-independent.
    """

    def __init__(self, A, T, kappa0=0.1, d_e=16, d_a=16, **kw):
        super().__init__(A, T, d_e=d_e, **kw)
        self.raw_kappa = nn.Parameter(torch.tensor(_inv_softplus(kappa0)))
        self.d_a = d_a
        self.att_emb = nn.Embedding(self.N, d_a)
        self.Wq = nn.Linear(d_a, d_a, bias=False)
        self.Wk = nn.Linear(d_a, d_a, bias=False)
        with torch.no_grad():
            self.Wq.weight.mul_(0.1)
            self.Wk.weight.mul_(0.1)

    def edge_multiplier(self):
        e = self.att_emb(self.idx)
        s = (self.Wq(e) @ self.Wk(e).T) / math.sqrt(self.d_a)
        s = 0.5 * (s + s.T)
        return 2.0 * torch.sigmoid(s)                                # (0, 2), =1 at s=0

    def coupling(self):
        return F.softplus(self.raw_kappa) * self.A * self.edge_multiplier()


# ── 4. Latent enforcement ────────────────────────────────────────────
class GNODElatent(GNODEBase):
    """Contiguity graph; L depends on (calendar month, unit [, frozen trend]).

    Fixes vs the old version:
      * seasonal features have a 12-month period (the old sin(2*pi*t_idx/T)
        had a period equal to the whole series, i.e. it was just a slow trend)
      * optional trend feature is frozen at its last training value, so test
        months are never extrapolated outside the training range
    """

    def __init__(self, A, T, kappa0=0.1, eps=0.01, d_e=16, use_trend=True, **kw):
        super().__init__(A, T, d_e=d_e, **kw)
        self.eps = eps
        self.use_trend = bool(use_trend)
        self.raw_kappa = nn.Parameter(torch.tensor(_inv_softplus(kappa0)))
        n_in = 2 + (1 if self.use_trend else 0) + d_e
        self.L_net = nn.Sequential(
            nn.Linear(n_in, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )

    def coupling(self):
        return F.softplus(self.raw_kappa) * self.A

    def _L_at(self, t_idx):
        dev = self.A.device
        ii = torch.as_tensor(t_idx, dtype=torch.long, device=dev).view(-1)
        tf = _season_feats(self.moy_all[ii], dev)                       # (M, 2)
        if self.use_trend:
            tr = ii.clamp(max=self.n_train - 1).float().view(-1, 1) / float(self.n_train)
            tf = torch.cat([tf, tr], dim=1)                             # (M, 3)
        M, nf = tf.shape
        feat = torch.cat([tf[:, None, :].expand(M, self.N, nf),
                          self.emb(self.idx)[None].expand(M, self.N, -1)], dim=2)
        out = torch.sigmoid(self.L_net(feat)).squeeze(-1)               # (M, N)
        return self.eps + (1.0 - self.eps) * out                        # [eps, 1)

    def enforcement(self):
        return self._L_at([self._t_idx])[0]

    @property
    def L(self):                                                        # (T, N)
        return self._L_at(list(range(self.T)))

    @property
    def L_all(self):
        return self.L.detach()

    def smoothness_loss(self, n=None):
        """Year-over-year change of L (same calendar month) over the first n
        months. Allows seasonality, penalises drift."""
        n = self.T if n is None else int(n)
        if n <= 12:
            return torch.zeros((), device=self.A.device)
        L = self._L_at(list(range(n)))
        return (L[12:] - L[:-12]).pow(2).mean()

    def get_learned_enforcement(self, nodes, months):
        L_np = self.L.detach().cpu().numpy()
        return pd.DataFrame([
            {"month": m, "region_id": n, "L_latent": float(L_np[t, i])}
            for t, m in enumerate(months) for i, n in enumerate(nodes)
        ])


MODEL_MAP = {
    "fc":           GNODEfc,
    "spatial":      GNODEspatial,
    "spatial_attn": GNODEattn,
    "latent":       GNODElatent,
}