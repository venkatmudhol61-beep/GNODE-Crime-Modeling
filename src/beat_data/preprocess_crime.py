import pandas as pd

def preprocess_crime():
    df = pd.read_csv("data/raw/crimes.csv", parse_dates=["Date"])
    df = df.dropna(subset=["Beat"])

    df["month"] = df["Date"].dt.to_period("M").astype(str)

    crime = (
        df.groupby(["Beat", "month"])
          .size()
          .reset_index(name="count")
    )

    crime["C"] = crime["count"] / crime["count"].max()

    crime.to_csv("data/processed/crime_beat_timeseries.csv", index=False)
    print("✅ crime_timeseries.csv created")

if __name__ == "__main__":
    preprocess_crime()
