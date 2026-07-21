import pandas as pd

print("🔎 Building consistent graph beat order (numeric sort)...")

# -----------------------------
# 1. Load data
# -----------------------------
crime = pd.read_csv("data/processed/crime_timeseries.csv")
geo = pd.read_csv("data/raw/beats.csv")

crime["Beat"] = crime["Beat"].astype(int)
geo["Beat"] = geo["BEAT_NUM"].astype(int)

# -----------------------------
# 2. Compute intersection
# -----------------------------
valid_beats = sorted(
    set(crime["Beat"].unique()) &
    set(geo["Beat"].unique())
)

print("Final graph beats:", len(valid_beats))

# -----------------------------
# 3. Save in numeric order
# -----------------------------
pd.Series(valid_beats).to_csv(
    "data/processed/graph_beats_order.csv",
    index=False
)

print("✅ graph_beats_order.csv saved in numeric order")
