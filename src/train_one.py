"""
src/train.py
============
Single training entry point for all four GN-ODE variants at both spatial
resolutions (district N=24, beat N=274) — using your original pasted model
classes unchanged (see model/ directory).

Usage
-----
    python src/train.py --resolution district --variant fc
    python src/train.py --resolution district --variant spatial
    python src/train.py --resolution district --variant attention
    python src/train.py --resolution district --variant latent
    python src/train.py --resolution beat      --variant fc
    python src/train.py --resolution beat      --variant spatial
    python src/train.py --resolution beat      --variant attention
    python src/train.py --resolution beat      --variant latent

Adjust the adjacency paths in RESOLUTION_CONFIG below to match your actual
data/processed/ layout.

Forward-signature differences are preserved exactly as in your pasted
scripts:
    fc, attention : model(t, C, L)      — unpacked C, observed L
    spatial        : model(t, y)         — packed (N,2) y
    latent          : model(t, y, t_idx) — packed (N,2) y, month index

Attention uses ONE model class (model/spatial_attention_gnode.py) at
both resolutions. It is a merged design built after two prior attempts
each failed at one resolution: a scalar-C-only model that collapsed at
N=274 (too little signal to discriminate neighbours once crime
intensities cluster), and a rich per-unit-embedding model that starved
and early-stopped at N=24 (its one N-scaling component got far less
gradient signal at low N). This version keeps per-unit alpha/beta
(proven safe at both N via GNODEBeat) and replaces the learned
embedding with each node's own graph degree — computed directly from
A, not learned, so it cannot be gradient-starved, while still giving
attention a real per-node signal to discriminate on. Verified via a
synthetic gradient-descent test that the design can learn genuinely
peaked (non-uniform) attention using degree alone at both N=24 and
N=274 — see conversation history for the test and results.
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchdiffeq import odeint

from model.fc_gnode_one import FCGNOde
from model.spatial_gnode_one import GNODE
from model.spatial_attention_gnode_one import SpatialAttentionGNODE
from model.latent_enforcement_gnode_one import LatentEnforcementGNODE

torch.manual_seed(42)
np.random.seed(42)

EPOCHS      = 200
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
ODE_METHOD  = "rk4"
LAMBDA_L    = 1e-3   # latent variant only — L2 reg on L_net

# ─────────────────────────────────────────────────────────────────────────────
# RESOLUTION CONFIG — only this differs between district and beat runs.
# Edit these paths to match your actual data/processed/ layout.
# ─────────────────────────────────────────────────────────────────────────────
RESOLUTION_CONFIG = {
    "district": dict(
        unit_col="District",
        crime_csv="data/processed/crime_timeseries.csv",
        arrest_csv="data/processed/arrest_timeseries.csv",
        adjacency_path="data/processed/adjacency.npy",
        order_csv=None,
    ),
    "beat": dict(
        unit_col="Beat",
        crime_csv="data/processed/crime_beat_timeseries.csv",
        arrest_csv="data/processed/arrest_beat_timeseries.csv",
        adjacency_path="data/processed/adjacency_beat.npy",
        order_csv="data/processed/graph_beats_order.csv",
    ),
}

# Per-variant hyperparameters, matching what your pasted scripts used.
VARIANT_HPARAMS = {
    "fc":        dict(district=dict(lr=1e-4, grad_clip=0.5, patience=20, sched_patience=10),
                       beat=    dict(lr=1e-4, grad_clip=0.5, patience=20, sched_patience=10)),
    "spatial":   dict(district=dict(lr=5e-4, grad_clip=1.0, patience=20, sched_patience=10),
                       beat=    dict(lr=1e-4, grad_clip=0.5, patience=20, sched_patience=10)),
    "attention": dict(district=dict(lr=5e-4, grad_clip=1.0, patience=20, sched_patience=10),
                       beat=    dict(lr=1e-4, grad_clip=0.5, patience=20, sched_patience=10)),
    "latent":    dict(district=dict(lr=5e-4, lr_l=5e-3, grad_clip=1.0, patience=20, sched_patience=10),
                       beat=    dict(lr=5e-4, lr_l=5e-3, grad_clip=1.0, patience=20, sched_patience=10)),
}

VARIANT_LABELS = {
    "fc": "Fully Connected GN-ODE",
    "spatial": "Spatial Graph GN-ODE",
    "attention": "Spatial Attention GN-ODE",
    "latent": "Latent Enforcement GN-ODE",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(resolution: str):
    cfg = RESOLUTION_CONFIG[resolution]
    unit_col = cfg["unit_col"]

    crime  = pd.read_csv(cfg["crime_csv"])
    arrest = pd.read_csv(cfg["arrest_csv"])
    crime.columns  = crime.columns.str.strip()
    arrest.columns = arrest.columns.str.strip()
    crime[unit_col]  = crime[unit_col].astype(int)
    arrest[unit_col] = arrest[unit_col].astype(int)

    if cfg["order_csv"] is not None:
        units = pd.read_csv(cfg["order_csv"]).iloc[:, 0].astype(int).tolist()
        crime  = crime[crime[unit_col].isin(units)].copy()
        arrest = arrest[arrest[unit_col].isin(units)].copy()
    else:
        units = sorted(crime[unit_col].unique().tolist())

    months = sorted(crime["month"].unique())
    A = np.load(cfg["adjacency_path"])

    if A.shape[0] != len(units):
        raise ValueError(
            f"Adjacency size {A.shape[0]} does not match unit count {len(units)} "
            f"for resolution='{resolution}'. Check RESOLUTION_CONFIG paths."
        )

    return crime, arrest, units, months, A, unit_col


def get_state(crime, arrest, units, unit_col, month):
    C = (crime[crime["month"] == month]
         .set_index(unit_col).reindex(units).fillna(0)["C"].values)
    L = (arrest[arrest["month"] == month]
         .set_index(unit_col).reindex(units).fillna(0)["L"].values)
    return (torch.tensor(C, dtype=torch.float32),
            torch.tensor(L, dtype=torch.float32))


def get_crime_state(crime, units, unit_col, month):
    C = (crime[crime["month"] == month]
         .set_index(unit_col).reindex(units).fillna(0)["C"].values)
    return torch.tensor(C, dtype=torch.float32)


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    smape = float(np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)
    return mae, rmse, r2, smape


def print_metrics(title, mae, rmse, r2, smape):
    bar = "=" * (len(title) + 10)
    print(f"\n{bar}\n===== {title} =====")
    print(f"  MAE   : {mae:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  R2    : {r2:.4f}")
    print(f"  sMAPE : {smape:.2f}%")
    print(f"{bar}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED TRAINING LOOP — identical structure/printing across all variants
# ─────────────────────────────────────────────────────────────────────────────
EMA_BETA = 0.7   # smoothing for the early-stop/checkpoint decision only —
                 # the raw val_loss is still what's logged and what the
                 # LR scheduler sees, this only affects "is this the best
                 # checkpoint" and "have we stopped improving" judgments.
                 # Matters most at low N (district): with only 24 units
                 # averaged per epoch, a single epoch's val_loss is noisy
                 # enough that an early lucky dip can look like the best
                 # checkpoint purely by chance, forcing every subsequent
                 # (genuinely improving) epoch to "beat" a fluke instead
                 # of a real trend. Smoothing removes that trap. At beat
                 # level (274 units, already smooth val curves) this
                 # changes little, since there's little noise to smooth.


def _run_training_loop(model, run_epoch, months_train, months_val, scheduler, opt, hp):
    best_val, best_smoothed, patience_c, history = float("inf"), None, 0, []
    best_state = None
    smoothed_val = None

    print("\n" + "=" * 58)
    print(f"{'Epoch':>6} {'Train':>12} {'Val':>12} {'Smoothed':>12} {'LR':>10}")
    print("=" * 58)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(months_train, train=True)
        val_loss   = run_epoch(months_val, train=False)
        lr_now = opt.param_groups[0]["lr"]

        smoothed_val = val_loss if smoothed_val is None else (
            EMA_BETA * smoothed_val + (1 - EMA_BETA) * val_loss
        )

        # Scheduler now reacts to the SAME smoothed signal as the
        # checkpoint/patience decision below. Previously it watched the
        # raw noisy val_loss, which meant the LR could get cut based on
        # noise-driven "no improvement" streaks even while the smoothed
        # trend was genuinely still improving — actively slowing down
        # the recovery this smoothing was supposed to allow.
        scheduler.step(smoothed_val)

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                         "val_smoothed": smoothed_val})
        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
              f"{smoothed_val:>12.6f}  {lr_now:>10.2e}")

        # Best-checkpoint and early-stop decisions use the SMOOTHED value —
        # this is the actual fix for noisy small-N validation curves.
        if best_smoothed is None or smoothed_val < best_smoothed:
            best_smoothed, best_val, patience_c = smoothed_val, val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"         >> best smoothed val {best_smoothed:.6f} — saved")
        else:
            patience_c += 1
            if patience_c >= hp["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    return best_val, history, best_state


# ─────────────────────────────────────────────────────────────────────────────
# UNPACKED-STATE PATH — fc, and attention at district level (model(t, C, L))
# ─────────────────────────────────────────────────────────────────────────────
def run_unpacked(model, crime, arrest, units, unit_col, months_train, months_val,
                  opt, scheduler, hp):
    loss_fn = nn.MSELoss()
    t_span  = torch.tensor([0.0, 1.0])

    def run_epoch(month_list, train=True):
        model.train() if train else model.eval()
        if train:
            opt.zero_grad()
        total_loss, pairs = 0.0, len(month_list) - 1
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in range(pairs):
                C0, L     = get_state(crime, arrest, units, unit_col, month_list[i])
                C_true, _ = get_state(crime, arrest, units, unit_col, month_list[i + 1])
                C_pred = odeint(
                    lambda t, C, l=L: model(t, C, l), C0, t_span, method=ODE_METHOD,
                )[-1]
                total_loss += loss_fn(C_pred, C_true)
            mean_loss = total_loss / pairs
            if train:
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
                opt.step()
        return mean_loss.item()

    return _run_training_loop(model, run_epoch, months_train, months_val, scheduler, opt, hp)


# ─────────────────────────────────────────────────────────────────────────────
# PACKED-STATE PATH — spatial (model(t, y)) — L held fixed via closure
# ─────────────────────────────────────────────────────────────────────────────
def run_spatial(model, crime, arrest, units, unit_col, months_train, months_val,
                 opt, scheduler, hp):
    loss_fn = nn.MSELoss()
    t_span  = torch.tensor([0.0, 1.0])

    def run_epoch(month_list, train=True):
        model.train() if train else model.eval()
        if train:
            opt.zero_grad()
        total_loss, pairs = 0.0, len(month_list) - 1
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in range(pairs):
                C0, L     = get_state(crime, arrest, units, unit_col, month_list[i])
                C_true, _ = get_state(crime, arrest, units, unit_col, month_list[i + 1])
                y0 = torch.stack([C0, L], dim=1)
                y_pred = odeint(lambda t, y: model(t, y), y0, t_span, method=ODE_METHOD)[-1]
                C_pred = y_pred[:, 0]
                total_loss += loss_fn(C_pred, C_true)
            mean_loss = total_loss / pairs
            if train:
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
                opt.step()
        return mean_loss.item()

    return _run_training_loop(model, run_epoch, months_train, months_val, scheduler, opt, hp)


# ─────────────────────────────────────────────────────────────────────────────
# UNPACKED-STATE PATH — attention at ANY resolution (model(t, C, L)).
# model.set_time_idx(i) is called before each odeint solve; it's a no-op
# at district scale and required internally at beat scale — this is what
# lets one model class serve both resolutions with one training path.
# ─────────────────────────────────────────────────────────────────────────────
def run_attention(model, crime, arrest, units, unit_col, months, month_to_idx,
                   months_train, months_val, opt, scheduler, hp):
    loss_fn = nn.MSELoss()
    t_span  = torch.tensor([0.0, 1.0])

    def run_epoch(month_list, train=True):
        model.train() if train else model.eval()
        if train:
            opt.zero_grad()
        total_loss, pairs = 0.0, len(month_list) - 1
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in range(pairs):
                m_now, m_next = month_list[i], month_list[i + 1]
                model.set_time_idx(month_to_idx[m_now])
                C0, L     = get_state(crime, arrest, units, unit_col, m_now)
                C_true, _ = get_state(crime, arrest, units, unit_col, m_next)
                C_pred = odeint(
                    lambda t, C, l=L: model(t, C, l), C0, t_span, method=ODE_METHOD,
                )[-1]
                total_loss += loss_fn(C_pred, C_true)
            mean_loss = total_loss / pairs
            if train:
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
                opt.step()
        return mean_loss.item()

    return _run_training_loop(model, run_epoch, months_train, months_val, scheduler, opt, hp)


# ─────────────────────────────────────────────────────────────────────────────
# PACKED-STATE PATH — latent (model(t, y, t_idx), t_idx drives internal MLP L)
# ─────────────────────────────────────────────────────────────────────────────
def run_latent(model, crime, units, unit_col, months, month_to_idx,
                months_train, months_val, optimizer, scheduler, hp):
    loss_fn = nn.MSELoss()
    t_span  = torch.tensor([0.0, 1.0])
    N = model.N

    def run_epoch(month_list, train=True):
        model.train() if train else model.eval()
        if train:
            optimizer.zero_grad()
        total_pred_loss, pairs = 0.0, len(month_list) - 1
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in range(pairs):
                m_now, m_next = month_list[i], month_list[i + 1]
                t_idx = month_to_idx[m_now]
                C0     = get_crime_state(crime, units, unit_col, m_now)
                C_true = get_crime_state(crime, units, unit_col, m_next)
                L_placeholder = torch.zeros(N)
                y0     = torch.stack([C0, L_placeholder], dim=1)
                y_pred = odeint(
                    lambda t, y, ti=t_idx: model(t, y, ti), y0, t_span, method=ODE_METHOD,
                )[-1]
                C_pred = y_pred[:, 0]
                total_pred_loss += loss_fn(C_pred, C_true)
            mean_pred_loss = total_pred_loss / pairs
            if train:
                l2_reg = LAMBDA_L * sum(p.pow(2).mean() for p in model.L_net.parameters())
                (mean_pred_loss + l2_reg).backward()
                nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
                optimizer.step()
        return mean_pred_loss.item()

    return _run_training_loop(model, run_epoch, months_train, months_val, scheduler, optimizer, hp)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GN-ODE trainer (district or beat)")
    parser.add_argument("--resolution", required=True, choices=["district", "beat"])
    parser.add_argument("--variant", required=True, choices=list(VARIANT_LABELS.keys()))
    args = parser.parse_args()

    label = f"{VARIANT_LABELS[args.variant]} ({args.resolution}-level)"
    print(f"Training {label}")

    crime, arrest, units, months, A, unit_col = load_data(args.resolution)
    N, T = len(units), len(months)
    print(f"Units  : {N}\nMonths : {T}\nAdjacency : {A.shape}")

    n_train = int(T * TRAIN_RATIO)
    n_val   = int(T * VAL_RATIO)
    months_train = months[:n_train]
    months_val   = months[n_train:n_train + n_val]
    month_to_idx = {m: i for i, m in enumerate(months)}
    print(f"Split : train={len(months_train)}  val={len(months_val)}  "
          f"test={T - n_train - n_val}")

    hp = VARIANT_HPARAMS[args.variant][args.resolution]

    checkpoint_path  = f"data/processed/{args.resolution}_{args.variant}_gnode_best.pt"
    history_path     = f"data/processed/{args.resolution}_{args.variant}_training_history.csv"
    predictions_path = f"data/processed/{args.resolution}_{args.variant}_predictions.csv"

    if args.variant == "fc":
        model = FCGNOde(N, alpha=0.3, beta=0.6)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=hp["sched_patience"], min_lr=1e-6)
        best_val, history, best_state = run_unpacked(model, crime, arrest, units, unit_col,
                                                        months_train, months_val, opt, scheduler, hp)

    elif args.variant == "spatial":
        model = GNODE(A, alpha=0.3, beta=0.6)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=hp["sched_patience"], min_lr=1e-6)
        best_val, history, best_state = run_spatial(model, crime, arrest, units, unit_col,
                                                       months_train, months_val, opt, scheduler, hp)

    elif args.variant == "attention":
        # T is only used internally when N > LARGE_N_THRESHOLD (beat scale);
        # harmless to always pass it.
        model = SpatialAttentionGNODE(A, hidden_dim=32, T=T)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=hp["sched_patience"], min_lr=1e-6)
        best_val, history, best_state = run_attention(model, crime, arrest, units, unit_col,
                                                         months, month_to_idx,
                                                         months_train, months_val,
                                                         opt, scheduler, hp)

    else:  # latent
        model = LatentEnforcementGNODE(A, T=T, hidden_dim=32)
        optimizer = torch.optim.Adam([
            {"params": [model.raw_alpha, model.raw_beta, model.gnn.weight], "lr": hp["lr"]},
            {"params": list(model.L_net.parameters()) + list(model.beat_emb.parameters()),
             "lr": hp["lr_l"]},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=hp["sched_patience"], min_lr=1e-6)
        best_val, history, best_state = run_latent(model, crime, units, unit_col, months,
                                                      month_to_idx, months_train, months_val,
                                                      optimizer, scheduler, hp)

    print(f"\nBest val loss: {best_val:.6f}")
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    model.eval()

    # ── Generate predictions on every month, tag split, compute test metrics ──
    months_set_train, months_set_val = set(months_train), set(months_val)
    t_span = torch.tensor([0.0, 1.0])
    results = []
    with torch.no_grad():
        for i in range(T - 1):
            m_now, m_next = months[i], months[i + 1]

            if args.variant in ("fc", "attention"):
                if args.variant == "attention":
                    model.set_time_idx(month_to_idx[m_now])
                C0, L = get_state(crime, arrest, units, unit_col, m_now)
                C_pred = odeint(lambda t, C: model(t, C, L), C0, t_span, method=ODE_METHOD)[-1]

            elif args.variant == "spatial":
                C0, L = get_state(crime, arrest, units, unit_col, m_now)
                y0 = torch.stack([C0, L], dim=1)
                y_pred = odeint(lambda t, y: model(t, y), y0, t_span, method=ODE_METHOD)[-1]
                C_pred = y_pred[:, 0]

            else:  # latent
                C0 = get_crime_state(crime, units, unit_col, m_now)
                t_idx = month_to_idx[m_now]
                L_placeholder = torch.zeros(N)
                y0 = torch.stack([C0, L_placeholder], dim=1)
                y_pred = odeint(lambda t, y: model(t, y, t_idx), y0, t_span, method=ODE_METHOD)[-1]
                C_pred = y_pred[:, 0]

            split_label = ("train" if m_next in months_set_train else
                           "val" if m_next in months_set_val else "test")
            for unit, value in zip(units, C_pred.numpy()):
                results.append({"month": m_next, unit_col: unit,
                                 "C_gnode": float(value), "split": split_label})

    pred_df = pd.DataFrame(results)
    pred_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")

    if args.variant == "latent":
        latent_df = model.get_learned_enforcement(units, months, id_col=unit_col)
        latent_path = f"data/processed/{args.resolution}_latent_enforcement.csv"
        latent_df.to_csv(latent_path, index=False)
        print(f"Saved learned L(t) to {latent_path}")

    test_pred = pred_df[pred_df["split"] == "test"].copy()
    real_df   = pd.read_csv(RESOLUTION_CONFIG[args.resolution]["crime_csv"])
    real_df.columns = real_df.columns.str.strip()
    real_df[unit_col] = real_df[unit_col].astype(int)
    real_df["month"] = real_df["month"].astype(str)
    test_pred["month"] = test_pred["month"].astype(str)
    merged = pd.merge(real_df, test_pred, on=["month", unit_col], how="inner")

    if merged.empty:
        print("WARNING: merge empty — check month column formats match.")
    else:
        mae, rmse, r2, smape = compute_metrics(merged["C"].values, merged["C_gnode"].values)
        print_metrics(f"{label.upper()} — TEST SET", mae, rmse, r2, smape)

    pd.DataFrame(history).to_csv(history_path, index=False)
    print("Training history saved.")


if __name__ == "__main__":
    main()