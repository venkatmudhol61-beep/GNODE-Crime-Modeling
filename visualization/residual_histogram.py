import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======================
# Load real data
# ======================
real = pd.read_csv("data/processed/crime_timeseries.csv")
real["month"] = pd.to_datetime(real["month"])

# ======================
# Load GNODE predictions
# ======================
gnode = pd.read_csv("data/processed/gnode_predictions.csv")
gnode["month"] = pd.to_datetime(gnode["month"])

# Safety
gnode["C_gnode"] = gnode["C_gnode"].clip(lower=0)

# ======================
# Merge on Beat + month
# ======================
df = real.merge(
    gnode,
    on=["Beat", "month"],
    how="inner"
)

# ======================
# Compute residuals
# residual = prediction - real
# ======================
df["residual"] = df["C_gnode"] - df["C"]

print(df["residual"].describe())

# ======================
# Plot histogram
# ======================
plt.figure(figsize=(8, 5))

plt.hist(
    df["residual"],
    bins=50,
    edgecolor="black",
    alpha=0.75
)

plt.axvline(
    0,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Zero Error"
)

plt.title("GNODE Residual Histogram (Prediction − Real)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
