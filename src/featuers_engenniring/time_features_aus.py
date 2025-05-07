# file: src/featuers_engenniring/time_features_aus.py

import pandas as pd
from pathlib import Path

def get_season(month: int) -> str:
    """Southern Hemisphere seasons."""
    if month in (9, 10, 11):
        return "Spring"
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    return "Winter"  # 6,7,8

def get_time_of_day(hour: int) -> str:
    """Custom buckets—you can tweak these boundaries if needed."""
    if 6 <= hour < 11:
        return "Morning"
    if 11 <= hour < 15:
        return "Noon"
    if 15 <= hour < 19:
        return "Afternoon"
    if 19 <= hour <= 23:
        return "Night"
    return "Other"  # e.g. 0–5

def main():
    # 1) define paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    input_csv  = PROJECT_ROOT / "data" / "raw"       / "melbourne" / "January_2019.csv"
    output_csv = PROJECT_ROOT / "data" / "processed" / "melbourne" / "melbourne_time_features.csv"

    # make sure output folder exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 2) load raw data
    df = pd.read_csv(input_csv)

    # 3) build datetime + staging features
    df["datetime"]     = pd.to_datetime(
        df["Date"] + " " + df["Hour"].astype(str) + ":00:00",
        format="%d/%m/%Y %H:%M:%S"
    )
    df["day"]          = df["datetime"].dt.date
    df["season"]       = df["datetime"].dt.month.apply(get_season)
    df["time_of_day"]  = df["datetime"].dt.hour.apply(get_time_of_day)

    # 4) unpivot: one row per (datetime, location, count)
    meta_cols    = ["datetime", "day", "season", "time_of_day"]
    sensor_cols  = [c for c in df.columns if c not in meta_cols + ["Date", "Hour"]]

    df_long = df.melt(
        id_vars=meta_cols,
        value_vars=sensor_cols,
        var_name="sensor",
        value_name="count"
    )

    # 5) (optional) drop NaNs or zeros
    df_long = df_long[df_long["count"].notna()]

    # 6) save fresh CSV
    df_long.to_csv(output_csv, index=False)
    print(f"✅ Wrote {len(df_long)} rows to {output_csv}")

if __name__ == "__main__":
    main()
