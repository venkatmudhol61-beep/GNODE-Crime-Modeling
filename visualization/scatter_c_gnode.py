import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Load real crime data (district)
# ======================
real = pd.read_csv("data/processed/crime_timeseries.csv")
real["month"] = pd.to_datetime(real["month"])

# ======================
# Load GNODE predictions (district)
# ======================
gnode = pd.read_csv("data/processed/gnode_predictions.csv")
gnode["month"] = pd.to_datetime(gnode["month"])

# ======================
# Merge on District + month
# ======================
df = real.merge(
    gnode,
    on=["District", "month"],
    how="inner"
)

# Safety lock
df["C_gnode"] = df["C_gnode"].clip(lower=0)

print("Merged rows:", len(df))

# ======================
# Scatter plot
# ======================
plt.figure(figsize=(6, 6))

plt.scatter(
    df["C"],
    df["C_gnode"],
    alpha=0.4,
    s=20
)

# 45-degree reference line
max_val = max(df["C"].max(), df["C_gnode"].max())
plt.plot([0, max_val], [0, max_val], "r--", linewidth=1)

plt.xlabel("Real Crime Intensity")
plt.ylabel("GNODE Predicted Crime")
plt.title("District-level GNODE: Prediction vs Real")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
