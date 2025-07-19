#!/usr/bin/env python3
"""
highway_features.py

Provides both a CLI `main()` for batch-processing sensor→highway CSV,
and a helper `compute_highway(gdf)` for on-the-fly use in the Flask API.
"""
import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- CLI file-path config ---
BASE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
SENS_CSV = os.path.join(BASE_DIR, "data", "raw", "melbourne", "sensor_osmid_lookup.csv")
GPKG = os.path.join(BASE_DIR, "data", "osm", "melbourne_street_network", "melbourne_network.gpkg")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "sensor_with_highway.csv")


def clean_highway(val):
    """
    Normalize a highway tag: pick first if list, split on ';' if string,
    else return as-is.
    """
    if isinstance(val, list) and val:
        return val[0]
    if isinstance(val, str) and ";" in val:
        return val.split(";", 1)[0]
    return val


def compute_highway(gdf, sensor_lookup=None):
    """
    Attach a 'highway' column to edges GeoDataFrame based on OSMIDs.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain 'osmid' column and optionally 'geometry'.
    sensor_lookup : GeoDataFrame or None
        If provided, must have columns ['sensor_name','osmid','geometry'] for fallback buffering.
        If None, buffering fallback is skipped and missing tags become 'unclassified'.

    Returns
    -------
    GeoDataFrame
        Input gdf with a new 'highway' column.
    """
    # Create a copy to avoid modifying the original
    gdf_work = gdf.copy()

    # Store original index for later restoration
    original_index = gdf_work.index.copy()

    # Normalize osmid column: convert lists to first element, ensure all are strings
    def normalize_osmid(osmid_val):
        if isinstance(osmid_val, list):
            return str(osmid_val[0]) if osmid_val else None
        return str(osmid_val) if osmid_val is not None else None

    # Create normalized osmid for merging
    gdf_work['osmid_normalized'] = gdf_work['osmid'].apply(normalize_osmid)

    # Create edges_attr with normalized osmid
    edges_attr = gdf_work[['osmid_normalized', 'highway']].drop_duplicates(subset='osmid_normalized', keep='first')

    # Perform merge on normalized osmid
    merged = gdf_work.merge(edges_attr, on='osmid_normalized', how='left', suffixes=('', '_direct'))
    merged['highway'] = merged['highway_direct']

    # Fill missing via fallback if sensor_lookup provided
    if sensor_lookup is not None:
        missing_mask = merged['highway'].isna()
        if missing_mask.any():
            # Buffer fallback: for each row without highway, take nearest sensor geometry
            buf = sensor_lookup.copy()
            buf = buf.to_crs(epsg=3857)
            sensor_sindex = buf.sindex
            edges_m = merged.to_crs(epsg=3857)

            def fallback_tag(osmid, geom):
                # query sensors by proximity
                possible = buf.iloc[list(sensor_sindex.intersection(geom.buffer(10).bounds))]
                if possible.empty:
                    return None
                # pick mode of their highway if present
                tags = possible['highway'].dropna()
                return tags.mode().iat[0] if not tags.empty else None

            for idx in merged[missing_mask].index:
                geom = merged.at[idx, 'geometry']
                tag = fallback_tag(merged.at[idx, 'osmid_normalized'], geom)
                merged.at[idx, 'highway'] = tag

    # Final fill
    merged['highway'] = merged['highway'].fillna('unclassified')

    # Clean up temporary columns and restore original structure
    result = merged.drop(columns=[c for c in merged.columns if c.endswith('_direct') or c == 'osmid_normalized'])

    # Ensure we maintain the original index order
    result = result.reindex(original_index)

    return result


def main():
    """
    CLI entrypoint: process all sensors and write sensor_with_highway.csv
    """
    logging.info(f"Loading sensors from: {SENS_CSV}")
    sensors = pd.read_csv(SENS_CSV, dtype=str, usecols=["sensor_name", "osmid"])
    sensors['sensor_name'] = sensors['sensor_name'].str.strip()
    sensors['osmid'] = sensors['osmid'].str.strip()

    logging.info(f"Reading edges layer from: {GPKG}")
    edges = gpd.read_file(GPKG, layer='edges')[['osmid', 'highway', 'geometry']]
    edges['osmid'] = edges['osmid'].astype(str)
    edges['highway'] = edges['highway'].apply(clean_highway)

    # Run compute_highway using fallback on sensors
    sensor_gdf = gpd.GeoDataFrame(
        sensors.merge(edges[['osmid', 'geometry']], on='osmid', how='left'),
        geometry='geometry', crs=edges.crs
    )
    result = compute_highway(edges, sensor_lookup=sensor_gdf)

    # Write CSV
    result[['osmid', 'highway']].to_csv(OUTPUT_CSV, index=False)
    logging.info(f"✅ Written {len(result)} rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()