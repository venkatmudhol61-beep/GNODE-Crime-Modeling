import os
import numpy as np
import pandas as pd
from shapely import wkt, affinity
from shapely.ops import unary_union
from sklearn.metrics.pairwise import haversine_distances

# ── Settings ──────────────────────────────────────────────────────────
CHICAGO_GEO = "data/raw/beats.csv"                  # the_geom (WKT), DISTRICT, BEAT_NUM
NYC_GEO     = "data/raw/nyc_precincts_geo.csv"      # Precinct, the_geom (WKT)

DATASETS = {
    "chicago_districts": {"order": "data/raw/graph_district_order.csv", "K": 2},
    "chicago_beats":     {"order": "data/raw/graph_beat_order.csv",     "K": 4},
    "nyc_precincts":     {"order": "data/raw/graph_precinct_order.csv", "K": 2},
}


def centroid_latlon(geom):
    """Area centroid in a local equirectangular projection, returned as (lat, lon)."""
    k = np.cos(np.radians(geom.centroid.y))
    c = affinity.scale(geom, xfact=k, yfact=1.0, origin=(0, 0)).centroid
    return c.y, c.x / k


# ── Load Chicago geometries (beats and districts) ─────────────────────
geo = pd.read_csv(CHICAGO_GEO)
geo["geom"]     = geo["the_geom"].apply(wkt.loads)
geo["BEAT_NUM"] = pd.to_numeric(geo["BEAT_NUM"], errors="coerce")
geo["DISTRICT"] = pd.to_numeric(geo["DISTRICT"], errors="coerce")

beat_geom = (
    geo.dropna(subset=["BEAT_NUM"])
       .assign(BEAT_NUM=lambda d: d["BEAT_NUM"].astype(int))
       .groupby("BEAT_NUM")["geom"]
       .apply(lambda s: unary_union(list(s)))
       .to_dict()
)
dist_geom = (
    geo.dropna(subset=["DISTRICT"])
       .assign(DISTRICT=lambda d: d["DISTRICT"].astype(int))
       .groupby("DISTRICT")["geom"]
       .apply(lambda s: unary_union(list(s)))
       .to_dict()
)

# ── Load NYC precinct geometries (CSV with WKT) ───────────────────────
nyc = pd.read_csv(NYC_GEO)
_cols = {c.lower(): c for c in nyc.columns}
_pc, _gc = _cols["precinct"], _cols["the_geom"]
nyc[_pc] = pd.to_numeric(nyc[_pc], errors="coerce")
nyc = nyc.dropna(subset=[_pc])
nyc_geom = (
    nyc.groupby(nyc[_pc].astype(int))[_gc]
       .apply(lambda s: unary_union([wkt.loads(x) for x in s]))
       .to_dict()
)

GEOMS = {
    "chicago_districts": dist_geom,
    "chicago_beats":     beat_geom,
    "nyc_precincts":     nyc_geom,
}

# ── Build adjacency for each dataset ──────────────────────────────────
for name, cfg in DATASETS.items():
    print(f"\n===== {name} =====")
    ids = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
    N, K = len(ids), cfg["K"]
    geoms = GEOMS[name]

    missing = [i for i in ids if i not in geoms]
    if missing:
        raise ValueError(
            f"{name}: regions without geometry: {missing}. "
            "Drop them from the graph-order file and rerun the later steps."
        )

    latlon = np.array([centroid_latlon(geoms[i]) for i in ids])
    dist   = haversine_distances(np.radians(latlon)) * 6371   # km

    # KNN, never using the node itself, then symmetrise
    k_act = min(K, N - 1)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        nbrs = [j for j in np.argsort(dist[i]) if j != i][:k_act]
        A[i, nbrs] = 1
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    assert np.array_equal(A, A.T), "Not symmetric!"

    # Symmetric normalisation D^-1/2 A D^-1/2
    deg    = A.sum(axis=1)
    d_inv  = np.where(deg > 0, deg ** -0.5, 0.0)
    A_norm = (np.diag(d_inv) @ A @ np.diag(d_inv)).astype(np.float32)

    # Fully connected adjacency, row-normalised
    A_fc = np.ones((N, N), dtype=np.float32)
    np.fill_diagonal(A_fc, 0)
    A_fc = (A_fc / A_fc.sum(axis=1, keepdims=True)).astype(np.float32)

    # Diagnostics
    kth = np.sort(dist, axis=1)[:, k_act]
    nb  = (A > 0).sum(axis=1)
    print(f"  N={N}  K={K}")
    print(f"  Distance to K-th neighbour (km): min {kth.min():.2f}  mean {kth.mean():.2f}  max {kth.max():.2f}")
    print(f"  Neighbours min/max/mean: {nb.min()} / {nb.max()} / {nb.mean():.2f}")
    print(f"  Nonzero entries: {int((A > 0).sum())}")
    print(f"  Symmetric: {np.allclose(A_norm, A_norm.T, atol=1e-6)}  |  Isolated nodes: {int((nb == 0).sum())}")

    os.makedirs("data/processed", exist_ok=True)
    out_sp = f"data/processed/{name}_adjacency_spatial.npy"
    out_fc = f"data/processed/{name}_adjacencyfc.npy"
    np.save(out_sp, A_norm)
    np.save(out_fc, A_fc)
    print(f"  Saved → {out_sp}")
    print(f"  Saved → {out_fc}")

print("\nAll 3 adjacency files built from boundary polygons.")