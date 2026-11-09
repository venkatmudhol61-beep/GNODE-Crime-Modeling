# Data Preprocessing

This step converts raw City of Chicago crime/arrest and NYC-Precincts records into the
normalized time series and spatial adjacency matrices used by all four
GNODE variants, at  beat-level (N=274), district-level (N=23), and NYC-Precincts
resolution.

## 1. Download the raw data

Download the following two files from the City of Chicago Open Data Portal
and place them in `data/raw/`:

- **Chicago Crime Data** (incident-level crime records):
%  https://data.cityofchicago.org/Public-Safety/City-of-Chicago-Crime-Data/v9q9-3dm2
  → save as `data/raw/crimes.csv`

- **Chicago Police Beat Boundaries** (beat boundary shapefiles/reference):
%  https://data.cityofchicago.org/Public-Safety/Boundaries-Police-Beats-current-/aerh-rz74
  → save as `data/raw/beats.csv`
simillarly other NYC data also.
The crime export must include, at minimum: `Date`, `Beat`, `District`, `NYC`,
`Latitude`, `Longitude`, `Arrest`.

## 2. What preprocessing does

For each spatial resolution (beat and district), the pipeline:

1. **Aggregates incidents into monthly counts** per spatial unit, from
   `Date` and the `Beat`/`District/NYC` column.
2. **Normalizes counts into [0,1]** by dividing each unit's raw monthly
   count by that unit's historical maximum count observed over the
   training period only (Jan 2001 – Jun 2018), per Eq. (3) of the paper.
   This produces the crime intensity series `C_i(t)` and, identically for
   arrest counts, the enforcement intensity series `L_i(t)`.
3. **Computes each unit's geographic centroid** as the mean latitude/
   longitude of all incidents recorded in that unit.
4. **Builds two adjacency matrices** from the centroids:
   - a **sparse spatial adjacency**, connecting two units if their
     centroids lie within a 6 km Haversine distance (Eq. 1–2 of the
     paper; sensitivity-checked over 4–10 km),
   - a **dense fully-connected adjacency** (all-ones, no self-loops), used
     by the Fully Connected variant.
5. **Writes a fixed unit ordering file** so that row/column `i` of every
   adjacency matrix always corresponds to the same spatial unit as row `i`
   of the time-series CSVs.

Beat-level and district-level data are processed independently from the
same source CSV (via the `Beat` and `District` columns respectively), so
the two resolutions constitute a controlled comparison rather than
different datasets.

## 3. Run preprocessing

```bash
python src/data/preprocess_crime.py  
```

## 4. Output files

Each run writes the following to `data/processed/`:

| File | Description |
|---|---|
| `crime_{res}_timeseries.csv` | Columns: `region_id, month, C` |
| `arrest_{res}_timeseries.csv` | Columns: `region_id, month, L` |
| `adjacency_{res}.npy` | Sparse geographic adjacency, shape (N, N) |
| `adjacencyfc.npy` | Dense fully-connected adjacency, shape (N, N) |
| `chicago_{res}_graph_order.csv` | Ordered list of `region_id`s matching adjacency rows/columns |

where `{res}` is `beats` (N=274) or `districts` (N=24).

## 5. Data splits

Months are split chronologically for all downstream training:

| Split | Months | Date range |
|---|---|---|
| Train | 1–210 | Jan 2001 – Jun 2018 |
| Validation | 211–240 | Jul 2018 – Dec 2020 |
| Test | 241–301 | Jan 2021 – Jan 2026 |

The historical-maximum normalization in step 2 uses **training months
only**, so no information from the validation or test period leaks into
the normalized `[0,1]` scale.

## 6. Training

Run the training script:

```bash
python src/train.py
```

This trains the model using the preprocessed data and saves the trained model and logs.

