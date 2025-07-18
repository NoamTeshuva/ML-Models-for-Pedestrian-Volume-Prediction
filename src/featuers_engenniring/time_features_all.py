#!/usr/bin/env python3
"""
time_features_all.py

Reads the merged raw CSV (merged_2019.csv), extracts datetime-based features,
and unpivots into one row per (datetime, sensor, count).  Writes out:
  • data/processed/melbourne/csv/melbourne_time_features.csv
"""
import pandas as pd
from pathlib import Path

def get_season(month: int) -> str:
    if month in (9,10,11): return "Spring"
    if month in (12,1,2):  return "Summer"
    if month in (3,4,5):   return "Autumn"
    return "Winter"

def get_time_of_day(hour: int) -> str:
    if 6 <= hour < 11:   return "Morning"
    if 11 <= hour < 15:  return "Noon"
    if 15 <= hour < 19:  return "Afternoon"
    if 19 <= hour <= 23: return "Night"
    return "Other"

def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_csv      = project_root / "data" / "raw" / "melbourne" / "merged_2019.csv"
    out_csv      = project_root / "data" / "processed" / "melbourne" / "csv" / "melbourne_time_features.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load full-year raw data
    df = pd.read_csv(raw_csv)

    # Build datetime + staging features
    # Assumes columns "Date" in dd/mm/YYYY and "Hour" integer plus one column per sensor
    # drop any trailing “.0” in the Hour field
    hours = df["Hour"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + hours + ":00:00",
        format="%d/%m/%Y %H:%M:%S",
        dayfirst=True
    )

    df["day"]         = df["datetime"].dt.date
    df["season"]      = df["datetime"].dt.month.map(get_season)
    df["time_of_day"] = df["datetime"].dt.hour.map(get_time_of_day)

    # Unpivot: one row per (datetime, sensor, count)
    meta_cols   = ["datetime", "day", "season", "time_of_day"]
    sensor_cols = [c for c in df.columns if c not in meta_cols + ["Date", "Hour"]]
    df_long = (
        df.melt(
            id_vars=meta_cols,
            value_vars=sensor_cols,
            var_name="sensor",
            value_name="count"
        )
        .dropna(subset=["count"])
    )

    # Write out
    df_long.to_csv(out_csv, index=False)
    print(f"✅ Wrote {len(df_long):,} rows to {out_csv}")

if __name__ == "__main__":
    main()
