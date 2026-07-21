import pandas as pd

def preprocess_arrest():
    df = pd.read_csv("data/raw/crimes.csv", parse_dates=["Date"])
    df = df.dropna(subset=["District"])

    df["month"] = df["Date"].dt.to_period("M").astype(str)

    arrest = (
        df.groupby(["District", "month"])["Arrest"]
          .mean()
          .reset_index(name="L")
    )

    arrest.to_csv("data/processed/arrest_timeseries.csv", index=False)
    print("✅ arrest_timeseries.csv created")

if __name__ == "__main__":
    preprocess_arrest()
