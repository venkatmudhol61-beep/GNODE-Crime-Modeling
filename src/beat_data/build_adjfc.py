import numpy as np
import pandas as pd

# Load crime data to get beats
crime = pd.read_csv("data/processed/crime_timeseries.csv")

# ✅ Load graph beat order to filter 305 → 274
graph_beats = (
    pd.read_csv("data/processed/graph_beats_order.csv")
    .iloc[:, 0]
    .astype(int)
    .tolist()
)

# ✅ Filter crime beats to only 274 graph beats
beats = [b for b in sorted(crime["Beat"].unique()) if b in set(graph_beats)]
N = len(beats)

print("Number of beats (graph):", N)                                     # 274

# Fully-connected adjacency (no self-loops)
A = np.ones((N, N), dtype=np.float32)
np.fill_diagonal(A, 0.0)

# Row-normalize (THIS is the key part)
A = A / (N - 1)

# Save
np.save("data/processed/adjacencyfc.npy", A)   # ✅ fc not adjacency.npy

print("Saved adjacency with shape:", A.shape)   # (274, 274)
print("Max value:", A.max())                    # 0.003663
print("Row sum (first row):", A[0].sum())       # 1.0