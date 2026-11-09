import pandas as pd
import unicodedata
import os

os.makedirs("data/processed", exist_ok=True)

TRAIN_RATIO  = 0.7
MONTHS       = [str(p) for p in pd.period_range("2006-01", "2024-12", freq="M")]
N_TRAIN      = int(len(MONTHS) * TRAIN_RATIO)
TRAIN_MONTHS = set(MONTHS[:N_TRAIN])

# "count": L = arrests per region-month / train-period max  (same definition as NYC)
# "rate" : L = fraction of crimes with arrest (call it an "arrest-rate proxy" in the paper)
CHICAGO_L_MODE = "count"

CONFIGS = {
    "chicago_districts": {
        "path": "data/raw/crimes.csv", "date_col": "Date", "region_col": "District",
        "arrest_col": "Arrest", "date_fmt": "mixed", "year_min": 2006, "year_max": 2024,
        "region_min": None, "region_max": None,
        "graph_order": "data/raw/graph_district_order.csv",
    },
    "chicago_beats": {
        "path": "data/raw/crimes.csv", "date_col": "Date", "region_col": "Beat",
        "arrest_col": "Arrest", "date_fmt": "mixed", "year_min": 2006, "year_max": 2024,
        "region_min": None, "region_max": None,
        "graph_order": "data/raw/graph_beat_order.csv",
    },
    "nyc_precincts": {
        "path": "data/raw/nyc_arrests.csv", "date_col": "ARREST_DATE", "region_col": "ARREST_PRECINCT",
        "arrest_col": None,                      # every row is an arrest -> counts
        "date_fmt": "%m/%d/%Y", "year_min": 2006, "year_max": 2024,
        "region_min": 1, "region_max": 123,
        "graph_order": "data/raw/graph_precinct_order.csv",
        # NYC arrest-file filters (decide these from the rules, not from test scores)
        "filters": {
            "dedupe_col":       "ARREST_KEY",          # one row per arrest
            "jurisdiction_col": "JURISDICTION_CODE",   # 0 patrol, 1 transit, 2 housing, >=3 non-NYPD
            "jurisdiction_max": 2,                     # keep NYPD only; None = keep all
            "law_cat_col":      "LAW_CAT_CD",          # F felony, M misdemeanor, V violation, I infraction
            "law_cats":         None,                  # e.g. ["F", "M"] to match the crime file; None = all
        },
    },
}

def normalize_id(val):
    if isinstance(val, str):
        return unicodedata.normalize("NFC", val.strip())
    return val

def parse_dates(s, fmt):
    if fmt == "mixed":
        out = pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        bad = out.isna()
        if bad.any():
            out[bad] = pd.to_datetime(s[bad], format="mixed", errors="coerce")
        return out
    return pd.to_datetime(s, format=fmt, errors="coerce")

def preprocess_arrest(dataset_name):
    cfg = CONFIGS[dataset_name]
    print(f"\nProcessing: {dataset_name}")

    df = pd.read_csv(cfg["path"], low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    needed = [c for c in (cfg["date_col"], cfg["region_col"], cfg["arrest_col"]) if c is not None]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise KeyError(f"{dataset_name}: missing {miss}. Available: {df.columns.tolist()}")

    flt = cfg.get("filters")
    if flt:
        n0 = len(df)
        k = flt.get("dedupe_col")
        if k and k in df.columns:
            df = df.drop_duplicates(subset=[k])
            print(f"  Dedupe on {k}: {n0:,} -> {len(df):,} rows")
        jc = flt.get("jurisdiction_col")
        if jc and jc in df.columns:
            code = pd.to_numeric(df[jc], errors="coerce")
            print(f"  {jc} counts: {code.value_counts().sort_index().head(8).to_dict()}")
            if flt.get("jurisdiction_max") is not None:
                df = df[code <= flt["jurisdiction_max"]]
        lc = flt.get("law_cat_col")
        if lc and lc in df.columns:
            print(f"  {lc} counts: {df[lc].value_counts().to_dict()}")
            if flt.get("law_cats"):
                df = df[df[lc].isin(flt["law_cats"])]
        print(f"  NYC filters kept {len(df):,} of {n0:,} rows")

    df[cfg["date_col"]] = parse_dates(df[cfg["date_col"]], cfg["date_fmt"])
    df = df.dropna(subset=[cfg["date_col"]])

    df = df[(df[cfg["date_col"]].dt.year >= cfg["year_min"]) &
            (df[cfg["date_col"]].dt.year <= cfg["year_max"])]

    df = df.dropna(subset=[cfg["region_col"]])
    if df[cfg["region_col"]].dtype == object:
        df[cfg["region_col"]] = df[cfg["region_col"]].apply(normalize_id)

    df["region_id"] = pd.to_numeric(df[cfg["region_col"]], errors="coerce")
    df = df.dropna(subset=["region_id"])
    df["region_id"] = df["region_id"].astype(int)

    if cfg["region_min"] is not None:
        df = df[df["region_id"] >= cfg["region_min"]]
    if cfg["region_max"] is not None:
        df = df[df["region_id"] <= cfg["region_max"]]

    graph_path = cfg.get("graph_order")
    if graph_path and os.path.exists(graph_path):
        valid_ids = set(pd.read_csv(graph_path).iloc[:, 0].astype(int).tolist())
        before = df["region_id"].nunique()
        df = df[df["region_id"].isin(valid_ids)]
        print(f"  Graph filter: {before} → {df['region_id'].nunique()} regions")
    else:
        print(f"  Graph filter: skipped (not found: {graph_path})")

    df["month"] = df[cfg["date_col"]].dt.to_period("M").astype(str)
    df = df[df["month"].isin(set(MONTHS))]

    use_count = (cfg["arrest_col"] is None) or (CHICAGO_L_MODE == "count")

    if use_count:
        if cfg["arrest_col"] is not None:        # Chicago: keep only rows that ended in arrest
            is_arrest = (df[cfg["arrest_col"]].astype(str).str.strip().str.lower()
                         .isin(["true", "1", "y", "yes"]))
            print(f"  Arrest rows kept: {is_arrest.sum():,} / {len(df):,}")
            df = df[is_arrest]
        arrest = df.groupby(["region_id", "month"]).size().reset_index(name="count")
        train_max = arrest.loc[arrest["month"].isin(TRAIN_MONTHS), "count"].max()
        if pd.isna(train_max) or train_max == 0:
            raise ValueError(f"{dataset_name}: train-period max arrests is 0 or NaN")
        arrest["L"] = arrest["count"] / train_max
        print(f"  L definition : normalised arrest count (train max = {train_max})")
    else:
        col = df[cfg["arrest_col"]]
        if col.dtype == object:
            col = (col.astype(str).str.strip().str.lower()
                      .map({"true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0}))
        df[cfg["arrest_col"]] = pd.to_numeric(col, errors="coerce")
        df = df.dropna(subset=[cfg["arrest_col"]])
        df[cfg["arrest_col"]] = df[cfg["arrest_col"]].astype(float)
        arrest = (df.groupby(["region_id", "month"])[cfg["arrest_col"]]
                    .mean().reset_index(name="L"))
        print("  L definition : arrest rate (proxy)")

    arrest = arrest[["month", "region_id", "L"]]
    out_path = f"data/processed/{dataset_name}_arrest_timeseries.csv"
    arrest.to_csv(out_path, index=False)

    print(f"  Regions : {arrest['region_id'].nunique()}")
    print(f"  Months  : {arrest['month'].nunique()}")
    print(f"  Range   : {arrest['month'].min()} → {arrest['month'].max()}")
    print(f"  L min/max/mean : {arrest['L'].min():.4f} / {arrest['L'].max():.4f} / {arrest['L'].mean():.4f}")
    print(f"  Rows    : {len(arrest):,}")
    print(f"  Saved  → {out_path}")

if __name__ == "__main__":
    for name in CONFIGS:
        preprocess_arrest(name)
    print("\nAll 3 datasets processed.")