"""
fig3_diagnostics_separate.py  (tight layout version)
  (a) gnode_scatter.png        predicted vs observed
  (b) gnode_residual_hist.png  residual histogram
Test-set predictions only, all units, no clipping.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET = "chicago_beats"
VARIANT = "latent"
TAG     = "v2"
SEED    = 42

PRED_FILE = f"data/processed/{DATASET}_{VARIANT}_{TAG}_seed{SEED}_predictions.csv"
REAL_FILE = f"data/processed/{DATASET}_crime_timeseries.csv"
PRED_COL  = "C_gnode"

def to_month(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()

real  = pd.read_csv(REAL_FILE)
gnode = pd.read_csv(PRED_FILE)
real["month"]  = to_month(real["month"])
gnode["month"] = to_month(gnode["month"])
if PRED_COL not in gnode.columns:
    sys.exit(f"'{PRED_COL}' not in {PRED_FILE}; columns: {gnode.columns.tolist()}")

test = gnode[gnode["split"] == "test"]
df = pd.merge(real[["region_id", "month", "C"]],
              test[["region_id", "month", PRED_COL]],
              on=["region_id", "month"], how="inner")
y, yhat = df["C"].values, df[PRED_COL].values
res = yhat - y

print(f"Units: {df['region_id'].nunique()}, test months: {df['month'].nunique()}, "
      f"predictions: {len(df)}")
print(f"Residual mean = {res.mean():.5f}, std = {res.std():.5f}")
print(f"MAE = {np.abs(res).mean():.4f}, RMSE = {np.sqrt((res**2).mean()):.4f}")
print(f"Outside [0,1]: {(yhat < 0).sum()} below 0, {(yhat > 1).sum()} above 1")

# ---------- (a) scatter: axes fitted to the 0.1%-99.9% range ----------
allv = np.concatenate([y, yhat])
lo = max(0.0, np.percentile(allv, 0.1) - 0.02)
hi = np.percentile(allv, 99.9) + 0.02
n_out = int(((y < lo) | (y > hi) | (yhat < lo) | (yhat > hi)).sum())
print(f"Points outside scatter axes: {n_out}")

fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.scatter(y, yhat, s=3, alpha=0.25, color="steelblue", rasterized=True)
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Observed crime intensity")
ax.set_ylabel("Predicted crime intensity")
ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(pad=0.4)
fig.savefig("gnode_scatter.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.show()

# ---------- (b) histogram: x-range fitted to 0.1%-99.9% of residuals ----------
rlo, rhi = np.percentile(res, [0.1, 99.9])
m = max(abs(rlo), abs(rhi))
n_out_r = int((np.abs(res) > m).sum())
print(f"Residuals outside histogram range: {n_out_r}")

fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.hist(res, bins=60, range=(-m, m), color="steelblue", edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Zero error")
ax.set_xlim(-m, m)
ax.set_xlabel(r"Residual $\hat C_i(t_k)-C_i(t_k)$")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(pad=0.4)
fig.savefig("gnode_residual_hist.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.show()