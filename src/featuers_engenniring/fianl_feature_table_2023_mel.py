#!/usr/bin/env python3
"""
create_feature_table_mlb.py

Reads all 12 monthly CSVs under data/raw/melbourne/2023,
melts them to long form (adding a `month` column!), merges with
static features (centrality, land use, highway), and writes one CSV:

  data/processed/melbourne/csv/feature_table_2023_melbourne.csv
"""
import os
import glob
import pandas as pd

# ────────────────────────────────────────────
# 1) CONFIG
# ────────────────────────────────────────────
YEAR       = "2023"
HERE       = os.path.dirname(__file__)
PROJECT    = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
RAW_DIR    = os.path.join(PROJECT, "data", "raw", "melbourne", YEAR)
STAT_DIR   = os.path.join(PROJECT, "data", "processed", "melbourne", "csv")
OUT_CSV    = os.path.join(STAT_DIR, f"feature_table_{YEAR}_melbourne.csv")
os.makedirs(STAT_DIR, exist_ok=True)

def month_to_season(m: int) -> int:
    if m in (12,1,2): return 1  # Winter
    if m in (3,4,5):  return 2  # Spring
    if m in (6,7,8):  return 3  # Summer
    return 4                 # Fall

def assign_time_of_day(h: int) -> int:
    if   6 <= h < 11:  return 1  # Morning
    if  11 <= h < 15:  return 2  # Afternoon
    if  15 <= h < 19:  return 3  # Evening
    return 0                   # Other

def normalize_label(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# ────────────────────────────────────────────
# 2) READ & MELT MONTHLY FILES
# ────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
print(f"Found {len(files)} files in {RAW_DIR!r}:")
for f in files:
    print("  ", os.path.basename(f))
if not files:
    raise FileNotFoundError("No monthly CSVs found!")

records = []
for path in files:
    base = os.path.basename(path)
    month = base.split("_")[0]        # e.g. "April"
    df   = pd.read_csv(path)
    print(f"→ {base}: raw rows = {len(df)}")

    # build datetime
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Hour"].astype(str) + ":00:00",
        format="%d/%m/%Y %H:%M:%S",
        dayfirst=True,
        errors="coerce"
    )

    # melt sensor columns
    keep = ["datetime"]
    sensors = [c for c in df.columns if c not in keep + ["Date","Hour"]]
    long = df.melt(
        id_vars=keep,
        value_vars=sensors,
        var_name="label",
        value_name="count"
    )

    # annotate source month
    long["month"] = month

    long["label"] = normalize_label(long["label"])
    long["count"] = pd.to_numeric(long["count"], errors="coerce")
    long = long.dropna(subset=["count"])
    records.append(long)

df_time = pd.concat(records, ignore_index=True).drop_duplicates(
    subset=["label","datetime","month","count"]
)
print("After concat:", len(df_time), "rows (should be sum of all!)")

# ────────────────────────────────────────────
# 3) TIME FEATURES
# ────────────────────────────────────────────
dt = df_time["datetime"]
df_time["Hour"]        = dt.dt.hour
df_time["Season"]      = dt.dt.month.map(month_to_season)
df_time["time_of_day"] = df_time["Hour"].apply(assign_time_of_day)
df_time["is_weekend"]  = dt.dt.dayofweek.isin([5,6]).astype(int)
df_time["Year"]        = int(YEAR)

df_time["volume_level"] = pd.qcut(
    df_time["count"], 5, labels=[1,2,3,4,5]
).astype(int)

# ────────────────────────────────────────────
# 4) LOAD STATIC FEATURES
# ────────────────────────────────────────────
# centrality (with human‐readable sensor_name)
df_cent = pd.read_csv(os.path.join(STAT_DIR, "sensor_centrality.csv"))
df_cent["sensor_name"] = df_cent["sensor_name"].astype(str).str.strip()
df_cent["label"]       = df_cent["sensor_name"].str.lower()
df_cent = df_cent[["sensor_name","label","betweenness","closeness"]].drop_duplicates()

# land use
df_lu = pd.read_csv(os.path.join(STAT_DIR, "sensor_landuse.csv"))
df_lu["label"] = df_lu["sensor_name"].astype(str).str.strip().str.lower()
df_lu = df_lu[["label","landuse"]].rename(columns={"landuse":"land_use"}).drop_duplicates()

# highway
df_hw = pd.read_csv(os.path.join(STAT_DIR, "sensor_with_highway.csv"))
df_hw["label"] = df_hw["sensor_name"].astype(str).str.strip().str.lower()
df_hw = df_hw[["label","highway"]].drop_duplicates()

df_static = (
    df_cent
    .merge(df_lu, on="label", how="left")
    .merge(df_hw, on="label", how="left")
)

# ────────────────────────────────────────────
# 5) MERGE & WRITE
# ────────────────────────────────────────────
df_final = df_time.merge(df_static, on="label", how="left")

# choose columns (including month!)
cols = [
    "sensor_name","label","month","datetime","count","Hour","Season",
    "time_of_day","is_weekend","volume_level",
    "betweenness","closeness","land_use","highway","Year"
]
df_final = df_final[cols]

# factorize categoricals
df_final["land_use"], lu_cats = pd.factorize(df_final["land_use"], sort=True)
df_final["highway"],  hw_cats = pd.factorize(df_final["highway"],  sort=True)

print("Land use cats:", lu_cats.tolist())
print("Highway cats: ", hw_cats.tolist())

df_final.to_csv(OUT_CSV, index=False)
print("✅ Wrote", OUT_CSV)
