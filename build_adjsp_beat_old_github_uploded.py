import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

# =========================
# SETTINGS
# =========================
CRIME_FILE = "data/raw/crimes.csv"
ORDER_FILE = "data/processed/graph_beats_order.csv"
OUT_FILE   = "data/processed/adjacency_beat.npy"

DIST_THRESHOLD_KM = 6   # good starting point
K = 4   # number of nearest neighbors (recommended: 5–10)
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

# ✅ Filter strictly to graph beats
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
# 4. Compute distance matrix
# =========================
coords_rad = np.radians(coords)
dist_matrix = haversine_distances(coords_rad) * 6371  # km

# =========================
# 5. Build KNN adjacency
# =========================
A = np.zeros((N, N), dtype=np.float32)

for i in range(N):
    # Get K nearest neighbors (excluding itself)
    idx = np.argsort(dist_matrix[i])[1:K+1]
    A[i, idx] = 1

# Symmetrize
A = np.maximum(A, A.T)

# =========================
# 6. Symmetric normalization (GNN standard)
# =========================
degree = A.sum(axis=1)
degree_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)

D_inv_sqrt = np.diag(degree_inv_sqrt)
A = D_inv_sqrt @ A @ D_inv_sqrt

# =========================
# 7. Save
# =========================
np.save(OUT_FILE, A)

# =========================
# 8. Diagnostics
# =========================
print("\n===== GRAPH STATS =====")
print("Shape:", A.shape)
print("Max value:", A.max())
print("Min value:", A.min())

neighbors = (A > 0).sum(axis=1)
print("Average neighbors:", neighbors.mean())
print("Min neighbors:", neighbors.min())
print("Max neighbors:", neighbors.max())

print("Is symmetric:", np.allclose(A, A.T, atol=1e-6))

# =========================
# 9. Connectivity check
# =========================
isolated = neighbors == 0

if isolated.any():
    print("⚠ WARNING: isolated nodes found")
else:
    print("✅ No isolated nodes")

print("\n✅ KNN spatial adjacency saved:", OUT_FILE)