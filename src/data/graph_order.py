"""Build graph-order files directly from the RAW crime files
(crimes.csv for Chicago, nyc_crimes.csv for NYC), not from processed timeseries.

Rule: a region is kept if
  1. it appears in the raw crime file within the study window,
  2. it exists in the boundary/geo file (Chicago only),
  3. it has at least one crime record in the TRAINING months.
Output: one CSV per dataset with a single column of sorted integer region ids.
"""
import os
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

START, END   = "2006-01", "2024-12"      # must match the window used for the processed files
TRAIN_RATIO  = 0.7
MIN_COVERAGE = None                      # e.g. 0.9 drops regions present in <90% of months
CHUNK        = 500_000                   # rows per chunk (crimes.csv is large)

MONTHS       = [str(p) for p in pd.period_range(START, END, freq="M")]
TRAIN_MONTHS = set(MONTHS[:int(len(MONTHS) * TRAIN_RATIO)])
ALL_MONTHS   = set(MONTHS)

# ---- CHECK THESE COLUMN NAMES AGAINST YOUR RAW FILE HEADERS ----------
CONFIGS = {
    "chicago_districts": {
        "raw_path": "data/raw/crimes.csv",
        "region_col": "District", "date_col": "Date",
        "date_fmt": "%m/%d/%Y %I:%M:%S %p",
        "out_path": "data/raw/graph_district_order.csv", "col_name": "district",
        "geo_path": "data/raw/beats.csv", "geo_col": "DISTRICT",
    },
    "chicago_beats": {
        "raw_path": "data/raw/crimes.csv",
        "region_col": "Beat", "date_col": "Date",
        "date_fmt": "%m/%d/%Y %I:%M:%S %p",
        "out_path": "data/raw/graph_beat_order.csv", "col_name": "beat",
        "geo_path": "data/raw/beats.csv", "geo_col": "BEAT_NUM",
    },
    "nyc_precincts": {
        "raw_path": "data/raw/nyc_crimes.csv",
        "region_col": "ADDR_PCT_CD", "date_col": "CMPLNT_FR_DT",
        "date_fmt": "%m/%d/%Y",
        "out_path": "data/raw/graph_precinct_order.csv", "col_name": "precinct",
        "geo_path": None, "geo_col": None,
    },
}


def scan_raw(cfg):
    """Return {region_id: set(months)} from the raw file, read in chunks."""
    seen = {}
    usecols = [cfg["region_col"], cfg["date_col"]]
    for chunk in pd.read_csv(cfg["raw_path"], usecols=usecols, chunksize=CHUNK,
                             low_memory=False):
        reg = pd.to_numeric(chunk[cfg["region_col"]], errors="coerce")
        dt = pd.to_datetime(chunk[cfg["date_col"]], format=cfg["date_fmt"],
                            errors="coerce")
        ok = reg.notna() & dt.notna()
        reg = reg[ok].astype(int)
        mon = dt[ok].dt.to_period("M").astype(str)
        part = pd.DataFrame({"r": reg.values, "m": mon.values}).drop_duplicates()
        part = part[part["m"].isin(ALL_MONTHS)]
        for r, m in zip(part["r"], part["m"]):
            seen.setdefault(r, set()).add(m)
    return seen


if __name__ == "__main__":
    scan_cache = {}   # Chicago beats/districts share one raw file: read it once
    for name, cfg in CONFIGS.items():
        key = (cfg["raw_path"], cfg["region_col"])
        if key not in scan_cache:
            print(f"  [{name}] scanning {cfg['raw_path']} ...")
            scan_cache[key] = scan_raw(cfg)
        seen = scan_cache[key]
        crime_ids = set(seen)

        if cfg["geo_path"] is not None:
            geo = pd.read_csv(cfg["geo_path"])
            geo_ids = set(pd.to_numeric(geo[cfg["geo_col"]], errors="coerce")
                          .dropna().astype(int).unique())
            valid_ids = sorted(crime_ids & geo_ids)
            dropped = crime_ids - geo_ids
            if dropped:
                print(f"  [{name}] dropped {len(dropped)} region(s) missing from geo: {sorted(dropped)}")
        else:
            valid_ids = sorted(crime_ids)

        # regions with no training-period crime cannot get a training-period centroid
        no_train = [r for r in valid_ids if not (seen[r] & TRAIN_MONTHS)]
        if no_train:
            print(f"  [{name}] dropped {len(no_train)} region(s) with no training-period data: {no_train}")
            valid_ids = [r for r in valid_ids if r not in set(no_train)]

        coverage = pd.Series({r: len(seen[r]) / len(MONTHS) for r in valid_ids})
        print(f"  [{name}] regions with <100% month coverage: {int((coverage < 1.0).sum())}")
        if MIN_COVERAGE is not None:
            low = set(coverage[coverage < MIN_COVERAGE].index)
            valid_ids = [r for r in valid_ids if r not in low]
            print(f"  [{name}] dropped {len(low)} region(s) below coverage {MIN_COVERAGE}")

        pd.DataFrame(valid_ids, columns=[cfg["col_name"]]).to_csv(cfg["out_path"], index=False)
        print(f"{os.path.basename(cfg['out_path'])} -> {len(valid_ids)} nodes")

    print("\nAll 3 graph order files created.")