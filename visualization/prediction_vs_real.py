"""
visualize_fc_gnode.py
=====================
Visualization for Fully Connected GN-ODE predictions.
All bugs fixed:
  1. Correct CSV filename (crime_timeseries.csv)
  2. Safe month alignment via pd.merge (not iloc)
  3. Burn-in applied AFTER merge (no offset issue)
  4. all_months cast to pandas Timestamp for axvline compatibility

Run from project root:
    python src/visualize_fc_gnode.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Load real data
# ======================
# FIX 1: correct filename — training uses crime_timeseries.csv
real = pd.read_csv("data/processed/crime_timeseries.csv")
real["month"] = pd.to_datetime(real["month"])

# ======================
# Load GNODE predictions
# ======================
gnode = pd.read_csv("data/processed/attention_gnode_predictions.csv")
gnode["month"] = pd.to_datetime(gnode["month"])

# Ensure non-negative predictions
gnode["C_gnode"] = gnode["C_gnode"].clip(lower=0)

# ======================
# Select district
# ======================
district_id = 1
BURN_IN     = 12

r = real[real["District"] == district_id].sort_values("month")
g = gnode[gnode["District"] == district_id].sort_values("month")

# FIX 2: merge on month so every row is guaranteed to align correctly.
# (Predictions are for months[i+1], so raw lengths differ by 1 — iloc was wrong)
merged = pd.merge(
    r[["month", "C"]],
    g[["month", "C_gnode", "split"]],
    on="month"
).sort_values("month").reset_index(drop=True)

# FIX 3: apply burn-in AFTER merge so both series share the same index
merged = merged.iloc[BURN_IN:].reset_index(drop=True)

# ======================
# Compute test split boundary
# FIX 4: cast to pandas Timestamp so axvline works correctly
# ======================
all_months = sorted(pd.to_datetime(real["month"].unique()))
T          = len(all_months)
n_train    = int(T * 0.7)
n_val      = int(T * 0.1)
val_start  = all_months[n_train]           # pandas Timestamp — val start
test_start = all_months[n_train + n_val]   # pandas Timestamp — safe for axvline

print(f"Districts in real data : {real['District'].nunique()}")
print(f"Total months           : {T}")
print(f"Train / Val / Test     : {n_train} / {n_val} / {T - n_train - n_val}")
print(f"Test starts at         : {test_start.date()}")
print(f"Rows after burn-in     : {len(merged)}")

# ======================
# FIGURE 1 — Trajectory
# ======================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    merged["month"], merged["C"],
    label="Real Crime",
    linewidth=1.5,
    alpha=0.7,
    color="steelblue",
)
ax.plot(
    merged["month"], merged["C_gnode"],
    label="FC GN-ODE Prediction",
    linewidth=2,
    color="darkorange",
)
ax.axvline(
    val_start,
    linestyle="--",
    color="green",
    alpha=0.8,
    label=f"Val Start ({val_start.date()})",
)
ax.axvline(
    test_start,
    linestyle="--",
    color="red",
    alpha=0.8,
    label=f"Test Start ({test_start.date()})",
)

ax.set_title(f"Spatial attention GN-ODE vs Real Crime — District {district_id}", fontsize=14)
ax.set_ylabel("Crime Intensity (Normalised)")
ax.set_xlabel("Time")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("trajectory_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: trajectory_plot.png")

# ======================
# FIGURE 2 — Residuals
# ======================
fig, ax = plt.subplots(figsize=(12, 4))

residual = merged["C_gnode"].values - merged["C"].values

ax.plot(merged["month"], residual, color="black", linewidth=1, label="Residual")
ax.axhline(0,          linestyle="--", color="grey",  alpha=0.6)
ax.axvline(
    val_start,
    linestyle="--",
    color="green",
    alpha=0.8,
    label=f"Val Start ({val_start.date()})",
)
ax.axvline(
    test_start,
    linestyle="--",
    color="red",
    alpha=0.8,
    label=f"Test Start ({test_start.date()})",
)

ax.set_title(f"Residuals (Pred − Real) — District {district_id}", fontsize=14)
ax.set_ylabel("Residual")
ax.set_xlabel("Time")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("residual_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: residual_plot.png")

# ======================
# FIGURE 3 — Scatter: Predicted vs Real
# ======================
fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(
    merged["C"], merged["C_gnode"],
    alpha=0.5,
    s=20,
    color="steelblue",
    label="Predictions",
)

# Perfect-prediction diagonal
lims = [
    min(merged["C"].min(), merged["C_gnode"].min()),
    max(merged["C"].max(), merged["C_gnode"].max()),
]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")

ax.set_title(f"Predicted vs Real — District {district_id}", fontsize=13)
ax.set_xlabel("Real Crime Intensity")
ax.set_ylabel("Predicted Crime Intensity")
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect("equal", "box")

plt.tight_layout()
plt.savefig("scatter_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: scatter_plot.png")

print("\n✅ All plots saved:")
print("   trajectory_plot.png")
print("   residual_plot.png")
print("   scatter_plot.png")