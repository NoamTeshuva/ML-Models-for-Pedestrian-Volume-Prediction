#!/usr/bin/env python3
"""
highway_features.py

Reads:
  - data/raw/melbourne/sensor_osmid_lookup.csv   (sensor_name → osmid)
  - data/osm/melbourne_street_network/melbourne_network.gpkg (edges + nodes)

For each sensor (including duplicates):
 1) Try a direct osmid → highway lookup in edges
 2) If still missing, fall back by buffering the node point by 10 m and
    picking the most common nearby highway tag
Writes:
  - data/processed/sensor_with_highway.csv
    (sensor_name, osmid, highway)
"""

import os
import logging
import pandas as pd
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR   = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
SENS_CSV   = os.path.join(BASE_DIR,
                          "data", "raw", "melbourne",
                          "sensor_osmid_lookup.csv")
GPKG       = os.path.join(BASE_DIR,
                          "data", "osm", "melbourne_street_network",
                          "melbourne_network.gpkg")
OUTPUT_CSV = os.path.join(BASE_DIR,
                          "data", "processed",
                          "sensor_with_highway.csv")

def clean_highway(val):
    if isinstance(val, list) and val:
        return val[0]
    if isinstance(val, str) and ";" in val:
        return val.split(";", 1)[0]
    return val

def main():
    # 1) Load sensors (keep duplicates)
    logging.info(f"Loading sensors from: {SENS_CSV}")
    sensors = pd.read_csv(SENS_CSV, dtype=str, usecols=["sensor_name", "osmid"])
    sensors["sensor_name"] = sensors["sensor_name"].str.strip()
    sensors["osmid"]        = sensors["osmid"].str.strip()

    # 2) Load edges (with highway & geometry)
    logging.info(f"Reading edges layer from: {GPKG}")
    edges = gpd.read_file(GPKG, layer="edges")[["osmid", "highway", "geometry"]]
    edges["osmid"]   = edges["osmid"].astype(str)
    edges["highway"] = edges["highway"].apply(clean_highway)

    # 3) Deduplicate edges for direct attribute lookup
    edges_attr = edges[["osmid", "highway"]].drop_duplicates(subset="osmid", keep="first")
    logging.info(f"→ edges_attr has {len(edges_attr)} unique osmids")

    # 4) Direct merge on osmid → highway
    logging.info("Performing direct merge of highway tags")
    df = sensors.merge(
        edges_attr,
        on="osmid",
        how="left",
        validate="many_to_one"
    )
    missing = df["highway"].isna().sum()
    logging.info(f"→ {missing} sensors missing after direct merge")

    # 5) Fallback for missing sensors
    if missing > 0:
        logging.info("Applying fallback via node buffering…")
        # 5a) Load nodes (to get each sensor's point)
        nodes = gpd.read_file(GPKG, layer="nodes")[["osmid", "geometry"]]
        nodes["osmid"] = nodes["osmid"].astype(str)

        # 5b) Attach node geometry to the missing sensors
        missing_df = df[df["highway"].isna()].merge(
            nodes, on="osmid", how="left"
        )
        gdf_miss = gpd.GeoDataFrame(missing_df, geometry="geometry", crs=nodes.crs)

        # 5c) Reproject both to metric CRS for buffering
        gdf_miss = gdf_miss.to_crs(epsg=3857)
        edges_m  = edges.to_crs(epsg=3857)

        # 5d) Buffer each sensor point by 10 m and spatial-join
        gdf_miss["buffer"] = gdf_miss.geometry.buffer(10)
        buf = gdf_miss.set_geometry("buffer")
        joined = gpd.sjoin(
            buf,
            edges_m[["highway", "geometry"]],
            how="left",
            predicate="intersects"
        )

        # 5e) Aggregate the edge-side highway tag (suffix '_right')
        fallback = (
            joined
            .dropna(subset=["highway_right"])
            .groupby("sensor_name")["highway_right"]
            .agg(lambda s: s.mode().iat[0])
            .rename("highway_fallback")
            .reset_index()
        )

        # 5f) Merge the fallback tags back and fill
        df = df.merge(fallback, on="sensor_name", how="left")
        df["highway"] = df["highway"].fillna(df["highway_fallback"])

    # 6) Final fill for any remaining
    df["highway"] = df["highway"].fillna("unclassified")

    # 7) Write out all rows (including duplicates)
    df[["sensor_name", "osmid", "highway"]].to_csv(OUTPUT_CSV, index=False)
    logging.info(f"✅ Written {len(df)} rows to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
