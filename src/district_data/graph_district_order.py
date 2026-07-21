import pandas as pd

print("🔎 Building consistent graph district order (numeric sort)...")

# -----------------------------
# 1. Load data
# -----------------------------
crime = pd.read_csv("data/processed/crime_timeseries_district.csv")
geo = pd.read_csv("data/raw/beats.csv")

crime["District"] = crime["District"].astype(int)
geo["District"] = geo["DISTRICT"].astype(int)

# -----------------------------
# 2. Compute intersection
# -----------------------------
valid_districts = sorted(
    set(crime["District"].unique()) &
    set(geo["District"].unique())
)

print("Final graph districts:", len(valid_districts))

# -----------------------------
# 3. Save in numeric order
# -----------------------------
pd.Series(valid_districts).to_csv(
    "data/processed/graph_district_order.csv",
    index=False
)

print("✅ graph_district_order.csv saved in numeric order")