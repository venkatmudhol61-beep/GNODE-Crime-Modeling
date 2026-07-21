import pandas as pd

def preprocess_crime():
    # Load raw crime data
    df = pd.read_csv(
        "data/raw/crimes.csv",
        parse_dates=["Date"]
    )

    # Keep valid districts only
    df = df.dropna(subset=["District"])

    # Convert to monthly time index
    df["month"] = df["Date"].dt.to_period("M").astype(str)

    # Aggregate crime counts per district per month
    crime = (
        df.groupby(["District", "month"])
          .size()
          .reset_index(name="count")
    )

    # Normalize crime intensity (C in GN-ODE)
    crime["C"] = crime["count"] / crime["count"].max()

    # Save processed data
    crime.to_csv(
        "data/processed/crime_timeseries.csv",
        index=False
    )

    print("✅ crime_timeseries.csv created")

if __name__ == "__main__":
    preprocess_crime()
 