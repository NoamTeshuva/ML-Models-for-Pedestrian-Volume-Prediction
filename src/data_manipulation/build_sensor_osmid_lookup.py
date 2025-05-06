"""
build_sensor_osmid_lookup.py

This script geocodes a list of sensor names using Nominatim,
finds the nearest OSM street segment for each one using your
Melbourne network, and saves a lookup table of sensor_name → osmid.

Inputs:
- sensor_list.txt (one name per line)
- melbourne_network.gpkg (edges layer)

Outputs:
- sensor_osmid_lookup.csv
"""

import os
import time
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------- paths ----------
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENSOR_TXT = os.path.join(BASE, "data", "raw", "melbourne", "sensor_list.txt")
GPKG = os.path.join(BASE, "data", "osm", "melbourne_street_network", "melbourne_network.gpkg")

OUT_CSV = os.path.join(BASE, "data", "raw", "melbourne", "sensor_osmid_lookup.csv")

# ---------- step 1: load sensor names ----------
with open(SENSOR_TXT, encoding="utf-8") as f:
    sensor_names = [line.strip() for line in f if line.strip()]

print(f"📍 Loaded {len(sensor_names)} sensor names")

# ---------- step 2: geocode via Nominatim ----------
geolocator = Nominatim(user_agent="pedestrian_volume_lookup")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

sensor_points = []
for name in tqdm(sensor_names, desc="Geocoding sensors"):
    try:
        result = geocode(name + ", Melbourne, Australia")
        if result:
            pt = Point(result.longitude, result.latitude)
            sensor_points.append((name, pt))
        else:
            print(f"❌ No geocode found for: {name}")
    except Exception as e:
        print(f"⚠️ Error for {name}: {e}")

gdf = gpd.GeoDataFrame(sensor_points, columns=["sensor_name", "geometry"], crs="EPSG:4326")
gdf = gdf.to_crs("EPSG:3857")  # match projection of the OSM network

# ---------- step 3: load OSM street edges ----------
edges = gpd.read_file(GPKG, layer="edges")
edges = edges.to_crs("EPSG:3857")

# ---------- step 4: spatial nearest join ----------
matched = gpd.sjoin_nearest(gdf, edges, how="left", distance_col="dist_meters")

# ---------- step 5: select columns + save ----------
lookup = matched[["sensor_name", "osmid", "dist_meters"]].copy()
lookup.to_csv(OUT_CSV, index=False)

print("✅ Sensor-to-osmid lookup saved to:", OUT_CSV)
