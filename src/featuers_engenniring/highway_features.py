"""
extract_highway_lookup.py

Builds a lookup of each sensor’s OSM highway tag (e.g. footway, tertiary, etc.).

Inputs:
 - sensor_osmid_lookup.csv   (sensor_name → osmid)
 - melbourne_network.gpkg    (edges layer with osmid & highway)

Output:
 - sensor_highway_lookup.csv (sensor_name, osmid, highway)
"""

import os
import pandas as pd
import geopandas as gpd

# ── Paths ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENSOR_LOOKUP = os.path.join(BASE, "data", "raw", "melbourne", "sensor_osmid_lookup.csv")
NETWORK_GPKG  = os.path.join(BASE, "data", "osm",   "melbourne_street_network", "melbourne_network.gpkg")
OUTPUT_CSV    = os.path.join(BASE, "data", "processed", "sensor_highway_lookup.csv")

# ── Load the sensor → osmid mapping ─────────────────────────────────────
sensors = pd.read_csv(SENSOR_LOOKUP, dtype={"sensor_name": str, "osmid": str})

# ── Load only the ‘osmid’ & ‘highway’ columns from the gpkg ───────────────
edges = gpd.read_file(NETWORK_GPKG, layer="edges")[["osmid", "highway"]]

# ── Normalize the highway field ───────────────────────────────────────────
def clean_highway(val):
    """
    OSMnx often returns a list or semicolon-separated string.
    We just take the first value in those cases.
    """
    if isinstance(val, list):
        return val[0]
    if isinstance(val, str) and ";" in val:
        return val.split(";", 1)[0]
    return val

edges["highway"] = edges["highway"].apply(clean_highway)

# ── Merge sensors with their highway tag ─────────────────────────────────
# sensors.osmid is string; edges.osmid may be int: coerce both to str
sensors["osmid"] = sensors["osmid"].astype(str)
edges["osmid"]   = edges["osmid"].astype(str)

lookup = sensors.merge(
    edges,
    on="osmid",
    how="left"
).drop_duplicates(["sensor_name"])

# ── Save ────────────────────────────────────────────────────────────────
lookup.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved sensor ⇢ highway lookup to: {OUTPUT_CSV}")
