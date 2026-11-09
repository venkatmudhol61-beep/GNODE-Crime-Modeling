import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchdiffeq import odeint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model.gnode import MODEL_MAP

# ─── CHANGE THESE LINES ONLY (or override via environment) ──────────
DATASET   = os.environ.get("DATASET", "chicago_districts")  # "chicago_beats" | "chicago_districts" | "nyc_precincts"
VARIANT   = os.environ.get("VARIANT", "spatial")            # "fc" | "spatial" | "spatial_attn" | "latent"
SEED      = int(os.environ.get("SEED", 42))
HIST      = int(os.environ.get("HIST", 4))                  # 1 = plain C_t
USE_TREND = os.environ.get("USE_TREND", "1") == "1"         # latent only
SEASONAL  = os.environ.get("SEASONAL", "1") == "1"          # seasonal growth rate on/off
TAG       = os.environ.get("TAG", "v2")
# ────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Training GN-ODE {TAG} | dataset={DATASET} variant={VARIANT} seed={SEED} hist={HIST}")

HPARAMS = {
    "chicago_beats"     : {"lr": 1e-3, "epochs": 300, "patience": 30},
    "chicago_districts" : {"lr": 1e-3, "epochs": 300, "patience": 30},
    "nyc_precincts"     : {"lr": 1e-3, "epochs": 300, "patience": 30},
}
hp       = HPARAMS[DATASET]
EPOCHS, LR, PATIENCE = hp["epochs"], hp["lr"], hp["patience"]

LR_L         = 5e-3
LAMBDA_L     = 5e-3         # year-over-year smoothness of latent L (training months only)
GRAD_CLIP    = 1.0          # matches the manuscript
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.1
BATCH_PAIRS  = 8
ODE_METHOD   = "rk4"
ODE_STEP     = 0.25
ODE_OPTS     = {"step_size": ODE_STEP}
SEASON       = 12
CHECKPOINT   = f"data/processed/{DATASET}_{VARIANT}_{TAG}_seed{SEED}_best.pt"
PREFIX       = f"data/processed/{DATASET}_{VARIANT}_{TAG}_seed{SEED}"

os.makedirs("data/processed", exist_ok=True)

CONFIGS = {
    "chicago_beats": {
        "crime": "data/processed/chicago_beats_crime_timeseries.csv",
        "arrest": "data/processed/chicago_beats_arrest_timeseries.csv",
        "order": "data/raw/graph_beat_order.csv",
        "adj_fc": "data/processed/chicago_beats_adjacencyfc.npy",
        "adj_sp": "data/processed/chicago_beats_adjacency_spatial.npy",
    },
    "chicago_districts": {
        "crime": "data/processed/chicago_districts_crime_timeseries.csv",
        "arrest": "data/processed/chicago_districts_arrest_timeseries.csv",
        "order": "data/raw/graph_district_order.csv",
        "adj_fc": "data/processed/chicago_districts_adjacencyfc.npy",
        "adj_sp": "data/processed/chicago_districts_adjacency_spatial.npy",
    },
    "nyc_precincts": {
        "crime": "data/processed/nyc_precincts_crime_timeseries.csv",
        "arrest": "data/processed/nyc_precincts_arrest_timeseries.csv",
        "order": "data/raw/graph_precinct_order.csv",
        "adj_fc": "data/processed/nyc_precincts_adjacencyfc.npy",
        "adj_sp": "data/processed/nyc_precincts_adjacency_spatial.npy",
    },
}
cfg = CONFIGS[DATASET]

# ── 1. LOAD DATA ────────────────────────────────────────────────────
crime   = pd.read_csv(cfg["crime"])
arrest  = pd.read_csv(cfg["arrest"])
regions = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
crime["region_id"]  = crime["region_id"].astype(int)
arrest["region_id"] = arrest["region_id"].astype(int)
months  = sorted(crime["month"].unique())
N, T    = len(regions), len(months)
print(f"Regions   : {N}\nMonths    : {T}")

# calendar month of every row (0 = Jan). Needed for the 12-month seasonality.
try:
    moy = [pd.Period(str(m), freq="M").month - 1 for m in months]
except Exception:
    print("WARNING: could not parse month labels; assuming the series starts in January")
    moy = [i % 12 for i in range(T)]

# ── 2. ADJACENCY ────────────────────────────────────────────────────
adj_file = cfg["adj_fc"] if VARIANT == "fc" else cfg["adj_sp"]
if not os.path.exists(adj_file):
    raise FileNotFoundError(f"Adjacency file not found: {adj_file}")
A_matrix = torch.tensor(np.load(adj_file), dtype=torch.float32)
print(f"Adjacency : {A_matrix.shape}  nonzero={(A_matrix > 0).sum().item()}")

# ── 3. MODEL ────────────────────────────────────────────────────────
if VARIANT not in MODEL_MAP:
    raise KeyError(f"Unknown VARIANT '{VARIANT}'. Choose from {list(MODEL_MAP)}")
extra = {"use_trend": USE_TREND} if VARIANT == "latent" else {}
model = MODEL_MAP[VARIANT](A_matrix, T=T, alpha=0.3, beta=0.6, hist=HIST,
                           seasonal=SEASONAL, **extra)
print(f"Model     : {model.__class__.__name__}")

# ── 4. OPTIMIZER ────────────────────────────────────────────────────
is_l = lambda n: n.startswith("L_net")
base_params = [p for n, p in model.named_parameters() if not is_l(n)]
lnet_params = [p for n, p in model.named_parameters() if is_l(n)]
groups = [{"params": base_params, "lr": LR}]
if lnet_params:
    groups.append({"params": lnet_params, "lr": LR_L})
opt = torch.optim.Adam(groups)

loss_fn   = nn.MSELoss()
t_span    = torch.tensor([0.0, 1.0])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── 5. DATA TENSORS ─────────────────────────────────────────────────
def build_matrix(df, col):
    pivot = df.pivot_table(index="month", columns="region_id", values=col, aggfunc="first")
    pivot = pivot.reindex(index=months, columns=regions)
    n_missing = int(pivot.isna().sum().sum())
    if n_missing:
        print(f"WARNING: {col}: {n_missing} missing (month, region) cells filled with 0")
    return pivot.fillna(0).values.astype(np.float32)

def build_windows(C, k):
    """(T, k, N): months t-k+1 .. t (oldest first), front-padded. Causal."""
    idx = (torch.arange(C.shape[0])[:, None] + torch.arange(-k + 1, 1)[None, :]).clamp(min=0)
    return C[idx]

C_all = torch.tensor(build_matrix(crime,  "C"))   # (T, N)
L_all = torch.tensor(build_matrix(arrest, "L"))   # (T, N)
assert C_all.sum() > 0, "C is all zeros - check region_id / month matching"
assert L_all.sum() > 0, "L is all zeros - check region_id / month matching"
print(f"C range   : [{C_all.min():.4f}, {C_all.max():.4f}]  mean={C_all.mean():.4f}")
print(f"L range   : [{L_all.min():.4f}, {L_all.max():.4f}]  mean={L_all.mean():.4f}")
if C_all.max() > 1.0 + 1e-6 or C_all.min() < -1e-6:
    print("NOTE: C leaves [0,1] after the training period (normalised by training max).")

Cwin_all = build_windows(C_all, HIST)             # (T, HIST, N)
MA3_all  = build_windows(C_all, 3).mean(dim=1)    # (T, N)  baseline only

# ── 6. SPLIT ────────────────────────────────────────────────────────
n_train      = int(T * TRAIN_RATIO)
n_val        = int(T * VAL_RATIO)
months_train = months[:n_train]
months_val   = months[n_train : n_train + n_val]
months_test  = months[n_train + n_val :]
print(f"Split     : train={len(months_train)}  val={len(months_val)}  test={len(months_test)}")
month_to_idx = {m: i for i, m in enumerate(months)}

model.set_calendar(moy, n_train=n_train)

def audit_print():
    if VARIANT == "latent":
        model.param_summary()
    else:
        model.param_summary(L_all[:n_train])

audit_print()

# ── 7. EPOCH FUNCTION ───────────────────────────────────────────────
def step_forward(C0):
    return odeint(model, C0, t_span, method=ODE_METHOD, options=ODE_OPTS)[-1]

def predict(gi):
    model.set_context(L_all[gi], t_idx=gi)        # GLOBAL month index
    C0 = model.init_state(Cwin_all[gi])           # smoothed, causal initial state
    return step_forward(C0)

def pair_loss(gi):
    return loss_fn(predict(gi), C_all[gi + 1])

def run_epoch(month_list, train=True):
    idxs = [month_to_idx[m] for m in month_list[:-1]]
    if len(idxs) < 1:
        return float("nan")
    if train:
        model.train()
        order = np.random.permutation(len(idxs))
        idxs  = [idxs[p] for p in order]
        total = 0.0
        for s in range(0, len(idxs), BATCH_PAIRS):
            batch = idxs[s : s + BATCH_PAIRS]
            opt.zero_grad()
            data_loss = sum(pair_loss(gi) for gi in batch) / len(batch)
            loss = data_loss
            if VARIANT == "latent":
                loss = loss + LAMBDA_L * model.smoothness_loss(n_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += data_loss.item() * len(batch)
        return total / len(idxs)
    model.eval()
    with torch.no_grad():
        return float(sum(pair_loss(gi) for gi in idxs).item() / len(idxs))

# ── 8. TRAINING LOOP ────────────────────────────────────────────────
# Reference: what persistence scores on the validation pairs (same MSE definition)
with torch.no_grad():
    vi = [month_to_idx[m] for m in months_val[:-1]]
    if vi:
        v_persist = float(np.mean([loss_fn(C_all[g], C_all[g + 1]).item() for g in vi]))
        print(f"Validation MSE of persistence (to beat): {v_persist:.6f}")

best_val, patience_c, history = float("inf"), 0, []
print("\n" + "=" * 58)
print(f"{'Epoch':>6} {'Train':>12} {'Val':>12} {'LR':>10}")
print("=" * 58)

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(months_train, train=True)
    val_loss   = run_epoch(months_val,   train=False)
    lr_now     = opt.param_groups[0]["lr"]
    scheduler.step(val_loss)
    history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
    print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {lr_now:>10.2e}")
    if val_loss < best_val:
        best_val, patience_c = val_loss, 0
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"         >> best val {best_val:.6f} - saved")
    else:
        patience_c += 1
        if patience_c >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nBest val loss : {best_val:.6f}")

# ── 9. PREDICTIONS ──────────────────────────────────────────────────
print("\nGenerating predictions from best checkpoint ...")
model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
model.eval()
audit_print()

set_train, set_val = set(months_train), set(months_val)
results = []
test_true, test_model, test_persist, test_ma3, test_season = [], [], [], [], []

with torch.no_grad():
    for i in range(T - 1):
        C_pred = predict(i)
        C0 = C_all[i]
        m = months[i + 1]
        split_label = "train" if m in set_train else "val" if m in set_val else "test"
        if split_label == "test":
            test_true.append(C_all[i + 1].numpy())
            test_model.append(C_pred.numpy())
            test_persist.append(C0.numpy())
            test_ma3.append(MA3_all[i].numpy())
            test_season.append(C_all[i + 1 - SEASON].numpy() if i + 1 - SEASON >= 0 else C0.numpy())
        for region, value, prev in zip(regions, C_pred.numpy(), C0.numpy()):
            results.append({"month": m, "region_id": region, "C_gnode": float(value),
                            "C_persist": float(prev), "split": split_label})

pd.DataFrame(results).to_csv(f"{PREFIX}_predictions.csv", index=False)
print(f"Saved -> {PREFIX}_predictions.csv")

# ── 10. TEST METRICS ────────────────────────────────────────────────
def metrics(y_true, y_pred):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    smape = float(np.mean(2 * np.abs(y_pred - y_true) /
                          (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)
    return mae, rmse, r2, smape

if len(test_true) == 0:
    print("WARNING: no test pairs found.")
else:
    y_true = np.concatenate(test_true).ravel()
    pm = np.concatenate(test_model).ravel()
    pp = np.concatenate(test_persist).ravel()
    print(f"\n===== {DATASET.upper()}  {VARIANT.upper()}  seed={SEED} - TEST (one-step-ahead) =====")
    for name, arr in [("GN-ODE", test_model), ("Persistence", test_persist),
                      ("Moving-avg-3", test_ma3), (f"Seasonal-naive-{SEASON}", test_season)]:
        mae, rmse, r2, smape = metrics(y_true, np.concatenate(arr).ravel())
        print(f"  [{name:<18}] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  sMAPE={smape:.2f}%")
    mse_m, mse_p = np.mean((y_true - pm) ** 2), np.mean((y_true - pp) ** 2)
    skill = 1.0 - mse_m / mse_p
    print(f"  Skill vs persistence (1 - MSE_model/MSE_persist): {skill:+.3f}  "
          f"-> {'BEATS' if skill > 0 else 'does NOT beat'} persistence")
    print(f"  Predictions outside [0,1] on test: {int(((pm < 0) | (pm > 1)).sum())} of {pm.size}")
    print("=" * 60)

# ── 11. LATENT ENFORCEMENT ANALYSIS ─────────────────────────────────
if VARIANT == "latent":
    print("\nGenerating learned enforcement analysis ...")
    with torch.no_grad():
        latent_df = model.get_learned_enforcement(regions, months)
        L_lat = model.L.detach().cpu().numpy()
    latent_df.to_csv(f"{PREFIX}_enforcement.csv", index=False)
    Lv = latent_df["L_latent"].values
    print(f"  Mean {Lv.mean():.4f}  Std {Lv.std():.4f}  Min {Lv.min():.4f}  Max {Lv.max():.4f}")
    C_np, L_obs = C_all.numpy(), L_all.numpy()
    print(f"  Corr(learned L, C)               : {np.corrcoef(L_lat.ravel(), C_np.ravel())[0,1]:.4f}")
    print(f"  Corr(learned L, observed arrests): {np.corrcoef(L_lat.ravel(), L_obs.ravel())[0,1]:.4f}")
    print(f"  Corr(mean learned L, time index) : {np.corrcoef(L_lat.mean(axis=1), np.arange(T))[0,1]:.4f}")
    print("  (learned L is a seasonal/trend suppression term, not a direct enforcement estimate)")

# ── 12. SAVE HISTORY ────────────────────────────────────────────────
pd.DataFrame(history).to_csv(f"{PREFIX}_training_history.csv", index=False)
print("History saved.")