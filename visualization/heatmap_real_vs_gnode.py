import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt

# =========================
# Load data
# =========================

crime = pd.read_csv("data/processed/crime_timeseries.csv")
gnode = pd.read_csv("data/processed/gnode_predictions.csv")
beats = pd.read_csv("data/raw/beats.csv")

# Convert month
crime["month"] = pd.to_datetime(crime["month"])
gnode["month"] = pd.to_datetime(gnode["month"])

# =========================
# Select month to visualize
# =========================
month_to_plot = crime["month"].max()

crime_m = crime[crime["month"] == month_to_plot]
gnode_m = gnode[gnode["month"] == month_to_plot]

# =========================
# Parse geometry → centroid
# =========================

beats["geometry"] = beats["the_geom"].apply(wkt.loads)
beats["lon"] = beats["geometry"].apply(lambda g: g.centroid.x)
beats["lat"] = beats["geometry"].apply(lambda g: g.centroid.y)

# =========================
# Merge data
# =========================

real_map = crime_m.merge(
    beats,
    left_on="Beat",
    right_on="BEAT_NUM",
    how="left"
)

gnode_map = gnode_m.merge(
    beats,
    left_on="Beat",
    right_on="BEAT_NUM",
    how="left"
)

# Safety clip
gnode_map["C_gnode"] = gnode_map["C_gnode"].clip(lower=0)

# =========================
# Plot
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# ---- Real Crime ----
sc1 = axes[0].scatter(
    real_map["lon"],
    real_map["lat"],
    c=real_map["C"],
    cmap="Reds",
    s=35
)
axes[0].set_title(f"Real Crime Intensity\n{month_to_plot.strftime('%Y-%m')}")
plt.colorbar(sc1, ax=axes[0], shrink=0.8)

# ---- GNODE Prediction ----
sc2 = axes[1].scatter(
    gnode_map["lon"],
    gnode_map["lat"],
    c=gnode_map["C_gnode"],
    cmap="Reds",
    s=35
)
axes[1].set_title(f"GNODE Prediction\n{month_to_plot.strftime('%Y-%m')}")
plt.colorbar(sc2, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()
