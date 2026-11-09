import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())

# ─── SETTINGS ────────────────────────────────────────────────────────
DATASET = "chicago_beats"      # chicago_beats | chicago_districts | nyc_precincts
SEED    = 42
TRAIN_RATIO, VAL_RATIO, LOOK_BACK = 0.7, 0.1, 4

CONFIGS = {
    "chicago_beats": dict(
        crime="data/processed/chicago_beats_crime_timeseries.csv",
        order="data/raw/graph_beat_order.csv",
        adj="data/processed/chicago_beats_adjacency_spatial.npy"),
    "chicago_districts": dict(
        crime="data/processed/chicago_districts_crime_timeseries.csv",
        order="data/raw/graph_district_order.csv",
        adj="data/processed/chicago_districts_adjacency_spatial.npy"),
    "nyc_precincts": dict(
        crime="data/processed/nyc_precincts_crime_timeseries.csv",
        order="data/raw/graph_precinct_order.csv",
        adj="data/processed/nyc_precincts_adjacency_spatial.npy"),
}
cfg = CONFIGS[DATASET]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════
# 1. DATA
# ═════════════════════════════════════════════════════════════════════
crime   = pd.read_csv(cfg["crime"])
regions = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
months  = sorted(crime["month"].unique())
N, T    = len(regions), len(months)

C_all = np.zeros((T, N), dtype=np.float32)
for i, m in enumerate(months):
    C_all[i] = (crime[crime["month"] == m]
                .set_index("region_id").reindex(regions)
                .fillna(0)["C"].values)

n_train = int(T * TRAIN_RATIO)
n_val   = int(T * VAL_RATIO)
train_t = list(range(LOOK_BACK, n_train))               # target months (train)
val_t   = list(range(n_train, n_train + n_val))         # target months (val)
test_t  = list(range(n_train + n_val, T))               # target months (test)
train_C = C_all[:n_train]

# Binary KNN graph recovered from the saved (normalised) adjacency
A_sp = (np.load(cfg["adj"]) > 0).astype(np.float32)
np.fill_diagonal(A_sp, 0)
A_row = A_sp / A_sp.sum(1, keepdims=True).clip(min=1.0)

def windows(target_idx):
    """Inputs are OBSERVED previous months: one-step-ahead, no rollout."""
    X = np.stack([C_all[t - LOOK_BACK:t] for t in target_idx]).astype(np.float32)
    Y = C_all[target_idx].astype(np.float32)
    return X, Y

X_tr, Y_tr = windows(train_t)
X_va, Y_va = windows(val_t)
X_te, Y_te = windows(test_t)
print(f"{DATASET}  N={N}  T={T}  train={len(train_t)}  "
      f"val={len(val_t)}  test={len(test_t)}  seed={SEED}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════════════
# 2. METRICS  (use this same function for GNODE)
# ═════════════════════════════════════════════════════════════════════
def metrics(y, p):
    y, p = y.ravel(), p.ravel()
    mae   = np.mean(np.abs(p - y))
    rmse  = np.sqrt(np.mean((p - y) ** 2))
    r2    = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
    smape = np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + 1e-8)) * 100
    return dict(MAE=mae, RMSE=rmse, R2=r2, sMAPE=smape)

# ═════════════════════════════════════════════════════════════════════
# 3. MODELS
# ═════════════════════════════════════════════════════════════════════
class SimpleLSTM(nn.Module):
    def __init__(self, n, hidden=64, layers=2):
        super().__init__()
        self.rnn = nn.LSTM(n, hidden, layers, batch_first=True,
                           dropout=0.1 if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, n)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

class SimpleGRU(nn.Module):
    def __init__(self, n, hidden=64, layers=2):
        super().__init__()
        self.rnn = nn.GRU(n, hidden, layers, batch_first=True,
                          dropout=0.1 if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, n)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=d, padding=self.pad)
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, k, d), nn.ReLU(),
            CausalConv1d(out_ch, out_ch, k, d), nn.ReLU())
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.net(x) + self.res(x)

class TCN(nn.Module):
    def __init__(self, n, hidden=64, n_layers=4):
        super().__init__()
        layers, ch = [], n
        for i in range(n_layers):
            layers.append(TCNBlock(ch, hidden, k=3, d=2 ** i))
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, n)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.fc(self.net(x)[:, :, -1])

class TransformerForecaster(nn.Module):
    def __init__(self, n, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                         dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.fc = nn.Linear(d_model, n)
    def forward(self, x):
        return self.fc(self.enc(self.proj(x))[:, -1])

class DCRNNCell(nn.Module):
    def __init__(self, N, in_d, hidden, k=2):
        super().__init__()
        self.N, self.hidden = N, hidden
        feat = (in_d + hidden) * (k + 1)
        self.gate = nn.Linear(feat, 2 * hidden)
        self.upd = nn.Linear(feat, hidden)
    def _diff(self, x, Als):
        out = [x]
        for A in Als:
            out.append(torch.einsum("ij,bjf->bif", A, out[-1]))
        return torch.cat(out, dim=-1)
    def forward(self, x, h, Als):
        B = x.size(0)
        xh = torch.cat([x, h], dim=-1)
        d = self._diff(xh, Als).reshape(B * self.N, -1)
        g = torch.sigmoid(self.gate(d))
        r, u = g.split(self.hidden, dim=1)
        xr = torch.cat([x.reshape(B * self.N, -1),
                        r * h.reshape(B * self.N, -1)], dim=1)
        xrd = self._diff(xr.reshape(B, self.N, -1), Als).reshape(B * self.N, -1)
        c = torch.tanh(self.upd(xrd))
        h_n = u * h.reshape(B * self.N, -1) + (1 - u) * c
        return h_n.reshape(B, self.N, self.hidden)

class DCRNN(nn.Module):
    def __init__(self, N, A, hidden=32, layers=2):
        super().__init__()
        At = torch.tensor(A, dtype=torch.float32)
        self.register_buffer("fw", At / At.sum(1, keepdim=True).clamp(min=1))
        self.register_buffer("bw", At.T / At.T.sum(1, keepdim=True).clamp(min=1))
        self.cells = nn.ModuleList([
            DCRNNCell(N, 1 if i == 0 else hidden, hidden) for i in range(layers)])
        self.hidden, self.N = hidden, N
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        B, L, N_ = x.shape
        Als = [self.fw, self.bw]
        hs = [torch.zeros(B, N_, self.hidden, device=x.device) for _ in self.cells]
        for t in range(L):
            inp = x[:, t].unsqueeze(-1)
            for i, cell in enumerate(self.cells):
                inp = cell(inp, hs[i], Als)
                hs[i] = inp
        return self.fc(hs[-1]).squeeze(-1)

class GWLayer(nn.Module):
    def __init__(self, ch, dil, n_sup):
        super().__init__()
        self.dil = dil
        self.f = nn.Conv2d(ch, ch, (1, 2), dilation=(1, dil))
        self.g = nn.Conv2d(ch, ch, (1, 2), dilation=(1, dil))
        self.mix = nn.Conv2d(ch * (n_sup + 1), ch, 1)
    def forward(self, x, sups):
        xp = F.pad(x, (self.dil, 0, 0, 0))                  # causal padding
        h = torch.tanh(self.f(xp)) * torch.sigmoid(self.g(xp))
        outs = [h] + [torch.einsum("vw,bcwt->bcvt", A, h) for A in sups]
        return x + self.mix(torch.cat(outs, 1))

class GraphWaveNet(nn.Module):
    """Compact Graph WaveNet: forward/backward supports + adaptive adjacency."""
    def __init__(self, N, A, hidden=32, emb=10):
        super().__init__()
        At = torch.tensor(A, dtype=torch.float32)
        self.register_buffer("fw", At / At.sum(1, keepdim=True).clamp(min=1))
        self.register_buffer("bw", At.T / At.T.sum(1, keepdim=True).clamp(min=1))
        self.E1 = nn.Parameter(torch.randn(N, emb) * 0.1)
        self.E2 = nn.Parameter(torch.randn(N, emb) * 0.1)
        self.start = nn.Conv2d(1, hidden, 1)
        self.layers = nn.ModuleList([GWLayer(hidden, d, 3) for d in (1, 2)])
        self.end = nn.Sequential(nn.ReLU(), nn.Conv2d(hidden, hidden, 1),
                                 nn.ReLU(), nn.Conv2d(hidden, 1, 1))
    def forward(self, x):                                   # x: (B, L, N)
        adp = torch.softmax(torch.relu(self.E1 @ self.E2.T), dim=1)
        sups = [self.fw, self.bw, adp]
        h = self.start(x.permute(0, 2, 1).unsqueeze(1))     # (B, C, N, L)
        for lyr in self.layers:
            h = lyr(h, sups)
        return self.end(h[..., -1:]).squeeze(-1).squeeze(1) # (B, N)

# ═════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ═════════════════════════════════════════════════════════════════════
def train_nn(model, lr=1e-3, epochs=200, patience=20, batch=16):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=8, factor=0.5, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.tensor(X_tr), torch.tensor(Y_tr)
    Xv, Yv = torch.tensor(X_va), torch.tensor(Y_va)
    best, best_w, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(Xt[b]), Yt[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), Yv).item()
        sched.step(vl)
        if vl < best:
            best, bad = vl, 0
            best_w = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_w)
    return model

def predict_nn(model):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_te)).numpy()

# ═════════════════════════════════════════════════════════════════════
# 5. DIEBOLD-MARIANO (one-step, per-month squared-error series)
# ═════════════════════════════════════════════════════════════════════
def dm_test(p1, p2, y):
    d = np.mean((p1 - y) ** 2, axis=1) - np.mean((p2 - y) ** 2, axis=1)
    var = np.var(d, ddof=1) / len(d)
    dm = d.mean() / np.sqrt(max(var, 1e-12))
    return float(dm), float(2 * (1 - stats.norm.cdf(abs(dm))))

# ═════════════════════════════════════════════════════════════════════
# 6. RUN
# ═════════════════════════════════════════════════════════════════════
P = {}
prev_idx = [t - 1 for t in test_t]

P["HistMean"] = np.tile(train_C.mean(0), (len(test_t), 1))   # train only
P["Naive"]    = C_all[prev_idx]                              # last observed month

ridge = Ridge(alpha=1.0).fit(C_all[:n_train - 1], C_all[1:n_train])
P["Ridge"] = ridge.predict(C_all[prev_idx])

try:
    mv  = VAR(train_C)
    lag = max(mv.select_order(maxlags=4).selected_orders.get("aic", 1), 1)
    res = mv.fit(lag, trend="c")
    P[f"VAR(lag={lag})"] = np.vstack(
        [res.forecast(C_all[t - lag:t], steps=1) for t in test_t])
except Exception as e:
    print(f"VAR failed ({e}); using VAR+PCA(20)")
    pca = PCA(n_components=min(20, N - 1)).fit(train_C)
    z = pca.transform(C_all)
    res = VAR(z[:n_train]).fit(1, trend="c")
    zf = np.vstack([res.forecast(z[t - 1:t], steps=1) for t in test_t])
    P["VAR+PCA(20)"] = pca.inverse_transform(zf)

zoo = {
    "LSTM":         (lambda: SimpleLSTM(N, 64, 2), 1e-3),
    "GRU":          (lambda: SimpleGRU(N, 64, 2), 1e-3),
    "TCN":          (lambda: TCN(N, 64, 4), 1e-3),
    "Transformer":  (lambda: TransformerForecaster(N, 64, 4, 2), 5e-4),
    "DCRNN":        (lambda: DCRNN(N, A_row, 32, 2), 5e-4),
    "GraphWaveNet": (lambda: GraphWaveNet(N, A_sp, 32), 5e-4),
}
params = {}
for name, (make, lr) in zoo.items():
    t0 = time.time()
    torch.manual_seed(SEED)
    model = train_nn(make(), lr=lr)
    P[name] = predict_nn(model)
    params[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name:<14} done in {time.time() - t0:.1f}s")

# ═════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═════════════════════════════════════════════════════════════════════
rows = [dict(Model=n, **metrics(Y_te, p)) for n, p in P.items()]
df = pd.DataFrame(rows).set_index("Model")
print("\n" + "=" * 65)
print(f"RESULTS - {DATASET.upper()}  seed={SEED}  test months={len(test_t)}")
print("=" * 65)
print(df.round(4).to_string())

print("\nDiebold-Mariano vs Naive (squared error, per month):")
for name, p in P.items():
    if name == "Naive":
        continue
    dm, pv = dm_test(p, P["Naive"], Y_te)
    print(f"  {name:<16} DM={dm:8.3f}  p={pv:.4f}  "
          f"{'sig' if pv < 0.05 else 'n.s.'}")

print("\nTrainable parameters:")
for n, c in params.items():
    print(f"  {n:<16} {c:>10,}")

os.makedirs("data/processed", exist_ok=True)
df.to_csv(f"data/processed/{DATASET}_baselines_seed{SEED}.csv")
np.savez(f"data/processed/{DATASET}_baseline_preds_seed{SEED}.npz", Y=Y_te, **P)
print(f"\nSaved -> data/processed/{DATASET}_baselines_seed{SEED}.csv")
print("Done.")