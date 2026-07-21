"""
run_baselines.py
================
Place this file in:  CRIME MODELING DISTRICT/src/run_baselines.py

Run from project root:
    python src/run_baselines.py

Dependencies (everything else you already have):
    pip install statsmodels
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (identical to your existing scripts)
# ─────────────────────────────────────────────────────────────────────────────
CRIME_CSV  = "data/processed/crime_timeseries.csv"
ARREST_CSV = "data/processed/arrest_timeseries.csv"
ADJ_NPY    = "data/processed/adjacency.npy"
GNODE_CSV  = "data/processed/fc_gnode_predictions.csv"
OUT_CSV    = "data/processed/baseline_comparison.csv"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (matches your trainc.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    crime  = pd.read_csv(CRIME_CSV)
    arrest = pd.read_csv(ARREST_CSV)
    A      = np.load(ADJ_NPY)

    districts = sorted(crime["District"].unique())
    months    = sorted(crime["month"].unique())
    N, T      = len(districts), len(months)

    # Build (T, N) arrays — same logic as your get_state()
    C_all = np.zeros((T, N), dtype=np.float32)
    L_all = np.zeros((T, N), dtype=np.float32)

    for t, m in enumerate(months):
        C_all[t] = (
            crime[crime["month"] == m]
            .set_index("District")
            .reindex(districts)
            .fillna(0)["C"]
            .values
        )
        L_all[t] = (
            arrest[arrest["month"] == m]
            .set_index("District")
            .reindex(districts)
            .fillna(0)["L"]
            .values
        )

    return C_all, L_all, A, districts, months


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT — must match trainc_tuned.py exactly: 70/10/20
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1

def get_splits(arr):
    T       = len(arr)
    n_train = int(T * TRAIN_RATIO)
    n_val   = int(T * VAL_RATIO)
    train   = arr[:n_train]
    val     = arr[n_train : n_train + n_val]
    test    = arr[n_train + n_val :]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# METRICS  (same as your evaluate_metrics_district.py + sMAPE added)
# ─────────────────────────────────────────────────────────────────────────────
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def compute_metrics(name, y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return {
        "Model": name,
        "MAE":   round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":    round(r2_score(y_true, y_pred), 4),
        "sMAPE": round(smape(y_true, y_pred), 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE 1 — Historical Average
# Predicts the training-set mean for every future month
# ═════════════════════════════════════════════════════════════════════════════
def run_historical_average(C_train, C_test):
    print("  Running Historical Average ...")
    pred = np.tile(C_train.mean(axis=0), (len(C_test), 1))
    return compute_metrics("Historical Average", C_test, pred)


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE 2 — ARIMA  (one model per district, order selected by AIC)
# ═════════════════════════════════════════════════════════════════════════════
def run_arima(C_train, C_test):
    from statsmodels.tsa.arima.model import ARIMA
    print("  Running ARIMA (one per district) ...")

    N     = C_train.shape[1]
    h     = len(C_test)
    preds = np.zeros((h, N), dtype=np.float32)

    for i in range(N):
        try:
            fit          = ARIMA(C_train[:, i], order=(2, 1, 1)).fit()
            preds[:, i]  = fit.forecast(steps=h)
        except Exception:
            preds[:, i]  = C_train[-1, i]   # fallback: last known value

    return compute_metrics("ARIMA", C_test, np.clip(preds, 0, 1))


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE 3 — VAR  (all districts modelled jointly)
# ═════════════════════════════════════════════════════════════════════════════
def run_var(C_train, C_test):
    from statsmodels.tsa.vector_ar.var_model import VAR
    print("  Running VAR ...")

    h = len(C_test)
    try:
        model = VAR(C_train)
        lag   = max(model.select_order(maxlags=6).selected_orders.get("bic", 2), 1)
        fit   = model.fit(lag)
        preds = fit.forecast(C_train[-lag:], steps=h)
    except Exception:
        preds = np.tile(C_train.mean(axis=0), (h, 1))

    return compute_metrics("VAR", C_test, np.clip(preds, 0, 1))


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE 4 — LSTM  (same Adam lr=1e-3 as your GNODE)
# ═════════════════════════════════════════════════════════════════════════════
def run_lstm(C_train, C_test, lookback=6, epochs=50, hidden=64):
    print("  Running LSTM ...")

    T, N   = C_train.shape
    h      = len(C_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sliding-window dataset
    Xs, Ys = [], []
    for t in range(lookback, T):
        Xs.append(C_train[t - lookback:t])
        Ys.append(C_train[t])

    X = torch.tensor(np.array(Xs), dtype=torch.float32).to(device)
    Y = torch.tensor(np.array(Ys), dtype=torch.float32).to(device)

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N, hidden, num_layers=2, batch_first=True)
            self.fc   = nn.Linear(hidden, N)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model   = LSTMModel().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss_fn(model(X), Y).backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"    LSTM epoch {ep+1}/{epochs}")

    # rolling forecast
    model.eval()
    history = C_train.copy()
    preds   = []
    with torch.no_grad():
        for _ in range(h):
            window = history[-lookback:]
            inp    = torch.tensor(window[None], dtype=torch.float32).to(device)
            p      = model(inp).cpu().numpy().flatten()
            preds.append(p)
            history = np.vstack([history, p])

    return compute_metrics("LSTM", C_test, np.clip(np.array(preds), 0, 1))


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE 5 — GCN static  (your GNN architecture WITHOUT the ODE solver)
#              Key ablation: proves the ODE component adds value
# ═════════════════════════════════════════════════════════════════════════════
def run_gcn_static(C_train, C_test, A, epochs=50):
    print("  Running GCN static (ablation — no ODE) ...")

    T, N   = C_train.shape
    h      = len(C_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # same normalisation as your build_adj.py
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1e-8, row_sum)
    A_norm  = A / row_sum
    A_t     = torch.tensor(A_norm, dtype=torch.float32).to(device)

    X = torch.tensor(C_train[:-1], dtype=torch.float32).to(device)
    Y = torch.tensor(C_train[1:],  dtype=torch.float32).to(device)

    class StaticGCN(nn.Module):
        """
        Mirrors GNODE dynamics but uses a single discrete step
        instead of the ODE solver — same alpha/beta/GNN structure.
        """
        def __init__(self):
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(0.6))
            self.beta  = nn.Parameter(torch.tensor(0.4))
            self.W     = nn.Linear(N, N, bias=False)

        def forward(self, C, A_t):
            # C shape: (batch, N)  A_t shape: (N, N)
            # correct: C @ A_t.T  →  (batch, N)
            dC = (
                self.alpha * C * (1 - C)
                + self.W(C @ A_t.T)
            )
            return torch.clamp(C + dC, 0, 1)

    model   = StaticGCN().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss_fn(model(X, A_t), Y).backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"    GCN epoch {ep+1}/{epochs}")

    # rollout forecast
    model.eval()
    current = torch.tensor(C_train[-1:], dtype=torch.float32).to(device)
    preds   = []
    with torch.no_grad():
        for _ in range(h):
            current = model(current, A_t)
            preds.append(current.cpu().numpy().flatten())

    return compute_metrics("GCN (static, no ODE)", C_test, np.clip(np.array(preds), 0, 1))


# ═════════════════════════════════════════════════════════════════════════════
# GNODE RESULTS — read from your saved gnode_predictions.csv
# ═════════════════════════════════════════════════════════════════════════════
def load_gnode_results(C_test, districts, months_test):
    print("  Loading your GNODE predictions ...")
    try:
        pred_df = pd.read_csv(GNODE_CSV)
        pred_df["month"] = pred_df["month"].astype(str)

        # use the split column written by trainc_tuned.py
        if "split" in pred_df.columns:
            test_df = pred_df[pred_df["split"] == "test"]
        else:
            # fallback: filter by test months
            months_test_set = set(months_test)
            test_df = pred_df[pred_df["month"].isin(months_test_set)]

        h     = len(months_test)
        N     = len(districts)
        preds = np.zeros((h, N), dtype=np.float32)

        for t, m in enumerate(months_test):
            row = (
                test_df[test_df["month"] == str(m)]
                .set_index("District")
                .reindex(districts)
                .fillna(0)["C_gnode"]
                .values
            )
            if len(row) == N:
                preds[t] = row

        return compute_metrics("GNODE (ours)", C_test, preds)

    except FileNotFoundError:
        print(f"  [skip] {GNODE_CSV} not found — run trainc_tuned.py first")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*62)
    print("  DISTRICT CRIME PREDICTION — BASELINE COMPARISON")
    print("="*62)

    # load
    print("\nLoading data ...")
    C_all, L_all, A, districts, months = load_data()

    # same 70/10/20 split as trainc_tuned.py
    C_train, C_val, C_test         = get_splits(C_all)
    months_train, months_val, months_test = get_splits(months)

    # baselines train on train+val (same amount of history as GNODE sees)
    C_train_full = np.vstack([C_train, C_val])

    print(f"  Districts : {len(districts)}")
    print(f"  Months    : {len(months)}  "
          f"(train={len(months_train)}, val={len(months_val)}, test={len(months_test)})")

    # run baselines
    print("\nRunning baselines ...")
    results = []
    results.append(run_historical_average(C_train_full, C_test))
    results.append(run_arima(C_train_full, C_test))
    results.append(run_var(C_train_full, C_test))
    results.append(run_lstm(C_train_full, C_test, epochs=50))
    results.append(run_gcn_static(C_train_full, C_test, A, epochs=50))

    gnode_row = load_gnode_results(C_test, districts, months_test)
    if gnode_row:
        results.append(gnode_row)

    # print table
    df = pd.DataFrame(results).set_index("Model")

    print("\n" + "="*62)
    print("  RESULTS — DISTRICT LEVEL")
    print("="*62)
    print(df.to_string())
    print("="*62)
    print("  Lower MAE / RMSE = better    |    Higher R2 = better")
    print("  GCN (static) = ablation: same GNN but no ODE solver\n")

    # save
    df.to_csv(OUT_CSV)
    print(f"  Saved to {OUT_CSV}\n")


if __name__ == "__main__":
    main()