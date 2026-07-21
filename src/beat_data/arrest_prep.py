import pandas as pd

def run():
    df = pd.read_csv("data/raw/crimes.csv", parse_dates=["Date"])
    df = df.dropna(subset=["Beat"])

    df["month"] = df["Date"].dt.to_period("M").astype(str)

    out = (
        df.groupby(["Beat", "month"])["Arrest"]
          .mean()
          .reset_index(name="L")
    )

    out.to_csv("data/processed/arrest_beat_timeseries.csv", index=False)
    print("arrest_ts.csv saved")

if __name__ == "__main__":
    run()
