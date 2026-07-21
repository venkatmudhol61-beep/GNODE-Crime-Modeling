import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import os

# =========================
# SETTINGS
# =========================
INPUT_FILE = "data/raw/crimes.csv"
OUTPUT_FILE = "data/processed/adjacency.npy"
DIST_THRESHOLD_KM = 6   # Try 5–8 km if needed

# =========================
# 1️⃣ Load CSV
# =========================
print("Loading CSV...")
df = pd.read_csv(INPUT_FILE)

# Check required columns
required_cols = ["District", "Latitude", "Longitude"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")

# Drop missing coordinates
df = df.dropna(subset=["Latitude", "Longitude"])

# =========================
# 2️⃣ Compute District Centroids
# =========================
print("Computing district centroids...")

centroids = (
    df.groupby("District")[["Latitude", "Longitude"]]
    .mean()
    .reset_index()
)

districts = centroids["District"].values
coords = centroids[["Latitude", "Longitude"]].values
N = len(districts)

print("Number of districts:", N)

# =========================
# 3️⃣ Compute Haversine Distance Matrix
# =========================
print("Computing distance matrix...")

coords_rad = np.radians(coords)
dist_matrix = haversine_distances(coords_rad) * 6371  # Earth radius km

# =========================
# 4️⃣ Build Spatial Adjacency
# =========================
print("Building adjacency...")

A = (dist_matrix < DIST_THRESHOLD_KM).astype(np.float32)

# Remove self loops
np.fill_diagonal(A, 0)

# Make symmetric
A = np.maximum(A, A.T)

# Replace steps 5 and 6 with this:

# =========================
# 5️⃣ Symmetric Normalization  D^{-1/2} A D^{-1/2}
# (matches manuscript equation and preserves symmetry)
# =========================
degree = A.sum(axis=1)
degree_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0)
D_inv_sqrt = np.diag(degree_inv_sqrt)
A = D_inv_sqrt @ A @ D_inv_sqrt

# =========================
# 6️⃣ Save
# =========================
os.makedirs("data/processed", exist_ok=True)
np.save(OUTPUT_FILE, A)

print("Adjacency saved.")
print("Shape:", A.shape)
print("Max value:", A.max())
print("Is symmetric:", np.allclose(A, A.T, atol=1e-6))
print("Average neighbors:", (A > 0).sum(axis=1).mean())