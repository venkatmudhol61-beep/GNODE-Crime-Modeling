"""
graphs.py
=========
Visualization for GN-ODE one-step-ahead predictions.
Reads the file written by the GN-ODE training script:
    data/processed/{DATASET}_{VARIANT}_{TAG}_seed{SEED}_predictions.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

# ─── MUST MATCH THE TRAINING RUN YOU WANT TO PLOT ───────────────────
DATASET = "nyc_precincts"
VARIANT = "fc"          # "fc" | "spatial" | "spatial_attn" | "latent"
TAG     = "v2"
SEED    = 42
REGION  = 1
BURN_IN = 12
# ────────────────────────────────────────────────────────────────────

PRED_FILE = f"data/processed/{DATASET}_{VARIANT}_{TAG}_seed{SEED}_predictions.csv"
REAL_FILE = f"data/processed/{DATASET}_crime_timeseries.csv"
PRED_COL  = "C_gnode"
LABEL     = f"Fully Connected GN-ODE"

def to_month(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()

# ======================
# Load data
# ======================
real  = pd.read_csv(REAL_FILE)
gnode = pd.read_csv(PRED_FILE)
real["month"]  = to_month(real["month"])
gnode["month"] = to_month(gnode["month"])

if PRED_COL not in gnode.columns:
    sys.exit(f"ERROR: '{PRED_COL}' not in {PRED_FILE}\n"
             f"Columns found: {gnode.columns.tolist()}\n"
             f"This looks like the wrong file (SDE output?). Check VARIANT/TAG/SEED.")

gnode[PRED_COL] = gnode[PRED_COL].clip(lower=0)

print(f"Real file : {REAL_FILE}")
print(f"Pred file : {PRED_FILE}")
print("Real columns:", real.columns.tolist())
print("Pred columns:", gnode.columns.tolist())

# ======================
# Select region and merge
# ======================
r = real[real["region_id"] == REGION].sort_values("month").reset_index(drop=True)
g = gnode[gnode["region_id"] == REGION].sort_values("month").reset_index(drop=True)
print(f"\nRows in real region {REGION}: {len(r)}")
print(f"Rows in pred region {REGION}: {len(g)}")

merged = (pd.merge(r[["month", "C"]], g[["month", PRED_COL, "split"]],
                   on="month", how="inner")
            .sort_values("month").reset_index(drop=True))
print(f"Merged rows BEFORE burn-in: {len(merged)}")

if len(merged) <= BURN_IN:
    sys.exit("ERROR: burn-in removes all rows. Set BURN_IN = 0 or choose another region.")
merged = merged.iloc[BURN_IN:].reset_index(drop=True)
print(f"Merged rows AFTER burn-in: {len(merged)}")

# ======================
# Split boundaries (same 70/10/20 rule as training)
# ======================
all_months = sorted(real["month"].unique())
T = len(all_months)
n_train = int(T * 0.7)
n_val   = int(T * 0.1)
val_start  = all_months[n_train]
test_start = all_months[n_train + n_val]

print("\n===== DATA SUMMARY =====")
print(f"Regions in real data : {real['region_id'].nunique()}")
print(f"Total months         : {T}")
print(f"Train / Val / Test   : {n_train} / {n_val} / {T - n_train - n_val}")
print(f"Val starts at        : {val_start.date()}")
print(f"Test starts at       : {test_start.date()}")

def add_split_lines(ax):
    ax.axvline(val_start, linestyle="--", color="green", alpha=0.8,
               label=f"Val Start ({val_start.date()})")
    ax.axvline(test_start, linestyle="--", color="red", alpha=0.8,
               label=f"Test Start ({test_start.date()})")

# ============================================================
# FIGURE 1 — Trajectory
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(merged["month"], merged["C"], label="Real Crime",
        linewidth=1.5, alpha=0.7, color="steelblue")
ax.plot(merged["month"], merged[PRED_COL], label=f"{LABEL} Prediction",
        linewidth=2, color="darkorange")
add_split_lines(ax)
ax.set_title(f"{LABEL} vs Real Crime — Region {REGION}", fontsize=14)
ax.set_ylabel("Crime Intensity (Normalized)")
ax.set_xlabel("Time")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("trajectory_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nSaved: trajectory_plot.png")

# ============================================================
# FIGURE 2 — Residuals
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
residual = merged[PRED_COL].values - merged["C"].values
ax.plot(merged["month"], residual, color="black", linewidth=1, label="Residual")
ax.axhline(0, linestyle="--", color="grey", alpha=0.6)
add_split_lines(ax)
ax.set_title(f"Residuals (Pred − Real) — Region {REGION}", fontsize=14)
ax.set_ylabel("Residual")
ax.set_xlabel("Time")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: residual_plot.png")

# ============================================================
# FIGURE 3 — Scatter (test months only)
# ============================================================
test_df = merged[merged["split"] == "test"]
if test_df.empty:
    print("No test rows for this region after burn-in; using all rows for scatter.")
    test_df = merged

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(test_df["C"], test_df[PRED_COL], alpha=0.5, s=20,
           color="steelblue", label="Test predictions")
lims = [min(test_df["C"].min(), test_df[PRED_COL].min()),
        max(test_df["C"].max(), test_df[PRED_COL].max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
ax.set_title(f"Predicted vs Real (test) — Region {REGION}", fontsize=13)
ax.set_xlabel("Real Crime Intensity")
ax.set_ylabel("Predicted Crime Intensity")
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect("equal", "box")
plt.tight_layout()
plt.savefig("scatter_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: scatter_plot.png")

print("\nAll plots saved: trajectory_plot.png, residual_plot.png, scatter_plot.png")