import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

print("Checking adjacency matrix...\n")

# ---------------------------
# Load adjacency matrix
# ---------------------------
A = np.load("data/processed/nyc_precincts_adjacency_spatial.npy")

# ---------------------------
# 1. Shape
# ---------------------------
print("Adjacency shape:", A.shape)

# ---------------------------
# 2. Row sum check
# ---------------------------
row_sums = A.sum(axis=1)
print("\nRow sum check:")
print("Min row sum:", row_sums.min())
print("Max row sum:", row_sums.max())

# ---------------------------
# 3. Node connection statistics
# ---------------------------
binary_A = (A > 0).astype(int)
degrees = binary_A.sum(axis=1)

print("\nNode connection statistics:")
print("Maximum node connections:", degrees.max())
print("Minimum node connections:", degrees.min())
print("Average node connections:", degrees.mean())

# ---------------------------
# 4. Symmetry check
# ---------------------------
print("\nSymmetry check:")
print("Is adjacency symmetric:", np.allclose(A, A.T))

# ---------------------------
# 5. Graph connectivity
# ---------------------------
G = nx.from_numpy_array(binary_A)

print("\nGraph information:")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Connected components:", nx.number_connected_components(G))

# ---------------------------
# 6. Visualize adjacency heatmap
# ---------------------------
plt.figure()
plt.imshow(A)
plt.title("Adjacency Matrix Heatmap")
plt.colorbar()
plt.show()

# ---------------------------
# 7. Visualize graph structure
# ---------------------------
plt.figure()
nx.draw(G, node_size=60)
plt.title("Spatial Graph Structure")
plt.show()