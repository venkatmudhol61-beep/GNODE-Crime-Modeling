import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import os

# =========================
# SETTINGS
# =========================
CRIME_FILE = "data/raw/crimes.csv"
ORDER_FILE = "data/processed/graph_beats_order.csv"
OUT_FILE   = "data/processed/adjacency_beat.npy"

DIST_THRESHOLD_KM = 6   # manuscript Eq. (2) / Section V-A
                         # sensitivity range tested: {4, 6, 8, 10} km

# =========================
# 1. Load graph beat order
# =========================
graph_beats = (
    pd.read_csv(ORDER_FILE)
    .iloc[:, 0]
    .astype(int)
    .tolist()
)
N = len(graph_beats)
print("Number of beats (graph):", N)   # should be 274

# =========================
# 2. Load raw crime data (WITH coords)
# =========================
df = pd.read_csv(CRIME_FILE)
df = df.dropna(subset=["Latitude", "Longitude", "Beat"])
df["Beat"] = df["Beat"].astype(int)
df = df[df["Beat"].isin(graph_beats)]
print("Unique beats after filter:", df["Beat"].nunique())

# =========================
# 3. Compute centroids (STRICT ORDER)
# =========================
centroids = (
    df.groupby("Beat")[["Latitude", "Longitude"]]
    .mean()
    .reindex(graph_beats)   # enforce order
)
coords = centroids.values
if coords.shape[0] != N:
    raise RuntimeError("Centroid mismatch!")

# =========================
# 4. Compute Haversine distance matrix (manuscript Eq. 1)
# =========================
coords_rad = np.radians(coords)
dist_matrix = haversine_distances(coords_rad) * 6371  # Earth radius km

# =========================
# 5. Build threshold adjacency (manuscript Eq. 2)
# =========================
print("Building threshold adjacency...")
A = (dist_matrix < DIST_THRESHOLD_KM).astype(np.float32)

# Remove self-loops (A_ii = 0, per manuscript)
np.fill_diagonal(A, 0)

# Symmetric by construction (Haversine distance is symmetric)
A = np.maximum(A, A.T)

# =========================
# 6. Symmetric normalization  W = D^{-1/2} A D^{-1/2}  (manuscript Eq. 4)
# =========================
degree = A.sum(axis=1)
degree_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
D_inv_sqrt = np.diag(degree_inv_sqrt)
A_norm = D_inv_sqrt @ A @ D_inv_sqrt

# =========================
# 7. Save
# =========================
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.save(OUT_FILE, A_norm)

# =========================
# 8. Diagnostics
# =========================
print("\n===== GRAPH STATS =====")
print("Shape:", A_norm.shape)
print("Max value:", A_norm.max())
print("Min value:", A_norm.min())

neighbors = (A > 0).sum(axis=1)   # raw (pre-normalisation) neighbour counts
print("Average neighbors:", neighbors.mean())
print("Min neighbors:", neighbors.min())
print("Max neighbors:", neighbors.max())
print("Is symmetric:", np.allclose(A_norm, A_norm.T, atol=1e-6))

# =========================
# 9. Connectivity check (manuscript requires every unit to have ≥1 neighbour)
# =========================
isolated = neighbors == 0
if isolated.any():
    isolated_beats = [graph_beats[i] for i in np.where(isolated)[0]]
    print(f"\n⚠ WARNING: {isolated.sum()} isolated node(s) found")
    print("Isolated beats:", isolated_beats)
    print("Consider raising DIST_THRESHOLD_KM or checking centroid data "
          "for these beats.")
else:
    print("\n✅ No isolated nodes")

print("\n✅ Threshold spatial adjacency saved:", OUT_FILE)