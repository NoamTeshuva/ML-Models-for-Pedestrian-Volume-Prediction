#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
landuse_features.py
===================
For each pedestrian sensor (with a chosen OSM way-ID), this script:
  1. Reads your existing 'sensor_osmid_lookup.csv' + 'osmid_locations.csv' to get
     sensor_name → OSMID + lon/lat.
  2. Loads your pre-downloaded land-use polygons from:
        data/processed/melbourne/gp/landuse_polygons.gpkg
     (EPSG:3857 or reprojects to EPSG:3857 as needed).
  3. Matches each sensor’s OSMID against the ‘edges’ layer in your existing
     melbourne_network.gpkg to retrieve the exact edge geometry.
  4. **Dissolves** all matched edge-segments into one MultiLineString per sensor.
  5. Buffers that dissolved geometry by BUFFER_METERS, then picks the single land-use
     polygon (only from {residential, retail, commercial}) whose centroid lies within
     that buffer.  If none of those three appear, assign None.
  6. Writes out:
       • sensor_paths.gpkg       – dissolved sensor-edge geometries (EPSG:3857)
       • sensor_landuse.gpkg     – sensor edges + chosen land-use label (EPSG:3857)
       • sensor_landuse.csv      – columns: sensor_id, sensor_name, landuse
"""

# ─────────── USER CONFIG ───────────
SENSOR_GEOM_CSV = r"C:\Users\HP\PycharmProjects\Pedestrian_Volume\data\processed\melbourne\csv\osmid_locations.csv"
SENSOR_MAP_CSV  = r"C:\Users\HP\PycharmProjects\Pedestrian_Volume\data\raw\melbourne\sensor_osmid_lookup.csv"

LANDUSE_GPKG    = r"C:\Users\HP\PycharmProjects\Pedestrian_Volume\data\processed\melbourne\gp\landuse_polygons.gpkg"
LANDUSE_LAYER   = "landuse_polygons"  # layer name inside the land-use GPKG

NETWORK_GPKG    = r"C:\Users\HP\PycharmProjects\Pedestrian_Volume\data\osm\melbourne_street_network\melbourne_network.gpkg"

OUTPUT_FOLDER   = r"C:\Users\HP\PycharmProjects\Pedestrian_Volume\data\processed\melbourne"
BUFFER_METERS   = 150
CRS_METRIC      = 3857  # EPSG:3857 for buffering / intersection
# ────────────────────────────────────

import sys
import re
import ast
from pathlib import Path

import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point

# ────────────── 1) LOAD SENSORS + CHOSEN OSMID ─────────────────
def read_sensors(geom_csv: Path, map_csv: Path) -> gpd.GeoDataFrame:
    """
    Read two CSVs:
      • geom_csv → columns: sensor_name, lon, lat  (may also have 'osmid', which we drop)
      • map_csv  → columns: sensor_name, osmid, dist_meters
    Returns a GeoDataFrame with columns: sensor_id, sensor_name, osmid, geometry (EPSG:4326).
    """
    # 1a) Load the geometry CSV
    df_geom = pd.read_csv(geom_csv)
    df_geom.columns = df_geom.columns.str.strip()

    # Must have sensor_name, lon, lat in the geometry file
    if "sensor_name" not in df_geom.columns:
        sys.exit(f"❌ ‘sensor_name’ not found in {geom_csv.name}.")
    low_cols = [c.lower() for c in df_geom.columns]
    if "lon" not in low_cols or "lat" not in low_cols:
        sys.exit(f"❌ ‘lon’ & ‘lat’ columns required in {geom_csv.name}.")

    # If geom_csv already had an 'osmid' column, drop it:
    if "osmid" in df_geom.columns:
        df_geom = df_geom.drop(columns=["osmid"])

    # Normalize sensor_name: strip whitespace + lowercase
    df_geom["sensor_name"] = (
        df_geom["sensor_name"].astype(str)
        .str.strip()
        .str.lower()
    )
    # Keep only the first occurrence of each sensor_name
    df_geom = df_geom.drop_duplicates("sensor_name", keep="first")

    # 1b) Load the mapping CSV (sensor_name, osmid, dist_meters)
    df_map = pd.read_csv(map_csv)
    df_map.columns = df_map.columns.str.strip()

    if "sensor_name" not in df_map.columns or "dist_meters" not in df_map.columns:
        sys.exit(f"❌ ‘sensor_name’ or ‘dist_meters’ not found in {map_csv.name}.")

    # Normalize sensor_name in mapping file the same way
    df_map["sensor_name"] = (
        df_map["sensor_name"].astype(str)
        .str.strip()
        .str.lower()
    )
    # Parse dist_meters as numeric, drop rows where parsing failed
    df_map["dist_meters"] = pd.to_numeric(df_map["dist_meters"], errors="coerce")
    df_map = df_map.dropna(subset=["sensor_name", "dist_meters"])

    # The second column in df_map (index=1) is assumed to be the OSMID column
    if len(df_map.columns) < 2:
        sys.exit(f"❌ Expected at least two columns (sensor_name + osmid) in {map_csv.name}.")
    osmid_col = df_map.columns[1]

    # Convert that column to numeric (some rows might be strings), then drop NA
    df_map[osmid_col] = pd.to_numeric(df_map[osmid_col], errors="coerce")
    df_map = df_map.dropna(subset=[osmid_col])
    df_map[osmid_col] = df_map[osmid_col].astype(int)

    # 1c) For each sensor_name, keep only the row with the *smallest* dist_meters
    idx = df_map.groupby("sensor_name")["dist_meters"].idxmin()
    df_best = (
        df_map.loc[idx, ["sensor_name", osmid_col]]
              .rename(columns={osmid_col: "osmid"})
    )

    # 2) Merge geometry + chosen OSMID on sensor_name
    df = df_geom.merge(df_best, on="sensor_name", how="inner")
    if df.empty:
        sys.exit(
            "❌ After merging, no sensor names matched between the two CSVs.\n"
            "   • Check that both files use exactly the same 'sensor_name' strings\n"
            "     (case-insensitive, no extra spaces).\n"
            "   • E.g. ‘City Square’ in one file must match ‘city square’ in the other."
        )

    # 3) Build a GeoDataFrame from lon/lat → geometry
    # Find the correct “lon” & “lat” columns (case-insensitive)
    lon_col = next(c for c in df.columns if re.match(r"^lon$", c, re.I))
    lat_col = next(c for c in df.columns if re.match(r"^lat$", c, re.I))

    gdf = gpd.GeoDataFrame(
        df.rename(columns={lon_col: "x", lat_col: "y"})[
            ["sensor_name", "osmid", "x", "y"]
        ],
        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
        crs=4326
    )
    # sensor_id duplicates sensor_name (all lowercase)
    gdf["sensor_id"] = gdf["sensor_name"]

    return gdf[["sensor_id", "sensor_name", "osmid", "geometry"]]


# ────────────── 2) MATCH SENSOR OSMIDs TO EDGES (FROM YOUR GPKG) ────────────────
def normalize_osm(x):
    """Normalize an edge’s osmid attribute into a Python list of ints."""
    if isinstance(x, list):
        return [int(i) for i in x if str(i).isdigit()]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return [int(i) for i in parsed if isinstance(i, (int, str)) and str(i).isdigit()]
            except Exception:
                inner = s[1:-1]
                parts = [p.strip() for p in inner.split(",")]
                return [int(p) for p in parts if p.isdigit()]
        if s.isdigit():
            return [int(s)]
        return []
    try:
        return [int(x)]
    except Exception:
        return []


def download_paths_by_osmid(sensor_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load edges from your local melbourne_network.gpkg, explode multi-OSMID entries,
    and return only those edges whose OSMID matches one of sensor_gdf['osmid'].
    The returned GeoDataFrame will include a 'sensor_id' column from sensor_gdf.
    """
    NETWORK_GPKG_PATH = Path(NETWORK_GPKG)
    if not NETWORK_GPKG_PATH.exists():
        raise FileNotFoundError(f"Cannot find network GPKG at {NETWORK_GPKG_PATH}")

    layers = set(fiona.listlayers(str(NETWORK_GPKG_PATH)))
    if "edges" not in layers:
        raise RuntimeError(f"'edges' layer not found in {NETWORK_GPKG_PATH} (found layers: {layers})")

    edges_gdf = gpd.read_file(str(NETWORK_GPKG_PATH), layer="edges")
    if edges_gdf.empty:
        return edges_gdf.copy()

    # Rename 'id' → 'osmid' if needed
    if "osmid" not in edges_gdf.columns:
        if "id" in edges_gdf.columns:
            edges_gdf = edges_gdf.rename(columns={"id": "osmid"})
        else:
            raise RuntimeError("The 'edges' layer has neither 'osmid' nor 'id' columns.")

    # Explode any multi-ID entries
    edges_reset = edges_gdf.reset_index(drop=True)[["u", "v", "key", "osmid", "geometry"]].copy()
    edges_reset["osmid_list"] = edges_reset["osmid"].apply(normalize_osm)
    edges_exp = edges_reset.explode("osmid_list").reset_index(drop=True)

    edges_exp["osmid"] = pd.to_numeric(edges_exp["osmid_list"], errors="coerce")
    edges_exp = edges_exp.dropna(subset=["osmid"]).copy()
    edges_exp["osmid"] = edges_exp["osmid"].astype(int)
    edges_exp = edges_exp.drop(columns=["osmid_list"])

    # Filter by the sensor-provided OSMIDs
    want_ids = set(sensor_gdf["osmid"].dropna().astype(int).tolist())
    mask = edges_exp["osmid"].isin(want_ids)
    filtered = edges_exp.loc[mask]

    if filtered.empty:
        empty = gpd.GeoDataFrame([], columns=edges_exp.columns, crs=edges_exp.crs)
        empty["sensor_id"] = pd.Series(dtype=str)
        return empty

    # Merge back to bring in sensor_id
    sensor_lookup = sensor_gdf[["sensor_id", "osmid"]].drop_duplicates()
    merged = filtered.merge(sensor_lookup, on="osmid", how="left")

    return gpd.GeoDataFrame(
        merged[["sensor_id", "osmid", "u", "v", "key", "geometry"]],
        geometry="geometry",
        crs=merged.crs,
    )


# ────────────── 3) ASSIGN LAND-USE BY NEAREST CENTROID (ONLY 3 TYPES) ────────────
def assign_landuse_by_nearest(
    edges: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    allowed=None
) -> pd.DataFrame:
    """
    For each row in `edges` (EPSG:3857), buffer by BUFFER_METERS and pick the land-use
    polygon (only from {residential, retail, commercial}) whose centroid lies within
    that buffer.  If none of those three appear inside, assign None.

    Returns a DataFrame with columns ['sensor_id', 'landuse'].
    """
    if allowed is None:
        allowed = {"residential", "retail", "commercial"}

    # 1) Compute centroids of every land-use polygon (EPSG:3857 assumed)
    land = land.copy()
    land["centroid"] = land.geometry.centroid

    # 2) Build a GeoDataFrame of centroids for spatial indexing
    centroids = gpd.GeoDataFrame({
        "landuse": land["landuse"],
        "geometry": land["centroid"]
    }, crs=land.crs)
    centroids_sindex = centroids.sindex

    out_rows = []
    for _, row in edges.iterrows():
        sid = row["sensor_id"]
        path_geom = row.geometry  # already in EPSG:3857

        # 3) Buffer the path by BUFFER_METERS
        buf_poly = path_geom.buffer(BUFFER_METERS)

        # 4) Find candidate centroids whose bounding-box intersects that buffer
        possible_index = list(centroids_sindex.intersection(buf_poly.bounds))
        if not possible_index:
            # no centroids even near the buffer → assign None
            out_rows.append({"sensor_id": sid, "landuse": None})
            continue

        candidates = centroids.iloc[possible_index].copy()
        # 5) Compute actual distance from each centroid to the buffer polygon
        candidates["dist_to_buffer"] = candidates.geometry.apply(
            lambda c: max(0.0, c.distance(buf_poly))
        )

        # 6) Filter centroids that lie strictly inside the buffer (distance == 0)
        inside_buf = candidates[candidates["dist_to_buffer"] == 0.0]
        if inside_buf.empty:
            # even though there were nearby centroids, none lie inside → assign None
            out_rows.append({"sensor_id": sid, "landuse": None})
            continue

        # 7) Among those inside buffer, restrict to allowed tags
        allowed_hits = inside_buf[inside_buf["landuse"].isin(allowed)]
        if allowed_hits.empty:
            # none of the allowed three appear inside → assign None
            out_rows.append({"sensor_id": sid, "landuse": None})
        else:
            # pick the allowed centroid that is “closest” to the buffer boundary
            best_idx = allowed_hits["dist_to_buffer"].idxmin()
            best = allowed_hits.loc[best_idx]
            out_rows.append({"sensor_id": sid, "landuse": best["landuse"]})

    return pd.DataFrame(out_rows)


# ────────────── MAIN FLOW ─────────────────────────────────────────────────────────
def main():
    csv_geom = Path(SENSOR_GEOM_CSV).resolve()
    csv_map  = Path(SENSOR_MAP_CSV).resolve()
    out_dir  = Path(OUTPUT_FOLDER).resolve()

    if not csv_geom.exists():
        sys.exit(f"❌ CSV not found: {csv_geom}")
    if not csv_map.exists():
        sys.exit(f"❌ CSV not found: {csv_map}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load sensors (EPSG:4326)
    sensors = read_sensors(csv_geom, csv_map)
    print(f"🔹 Loaded {len(sensors)} unique sensors")

    # 2) Build study-area polygon to ensure we load the correct subset (if needed)
    poly_m = sensors.to_crs(CRS_METRIC).geometry.unary_union.convex_hull.buffer(BUFFER_METERS * 3)
    poly_ll = gpd.GeoSeries([poly_m], crs=CRS_METRIC).to_crs(4326).iloc[0]

    # 3) Load pre-downloaded land-use polygons (EPSG:3857)
    land_path = Path(LANDUSE_GPKG).resolve()
    if not land_path.exists():
        sys.exit(f"❌ landuse_polygons.gpkg not found at {land_path}")
    land = gpd.read_file(str(land_path), layer=LANDUSE_LAYER)
    # Ensure it's in EPSG:3857
    if land.crs is None or land.crs.to_epsg() != CRS_METRIC:
        land = land.to_crs(CRS_METRIC)
    # Keep only Polygon/MultiPolygon & 'landuse'
    land = land[land.geometry.type.isin({"Polygon", "MultiPolygon"})].copy()
    if "landuse" not in land.columns:
        sys.exit(f"❌ ‘landuse’ column not found in layer ‘{LANDUSE_LAYER}’ of {land_path.name}.")
    land = land[["landuse", "geometry"]]
    print(f"🔹 Loaded {len(land):,} land-use polygons from {land_path.name}")

    # 4) Load sensor edges by OSMID (EPSG:4326 → reproject to EPSG:3857)
    edges = download_paths_by_osmid(sensors)
    if edges.empty:
        print("⚠️ No sensor edges were matched by OSMID → exiting.")
        return

    edges_3857 = edges.to_crs(CRS_METRIC)

    # ── 4a) Dissolve all edge-segments that share the same sensor_id into one MultiLineString
    edges_single = (
        edges_3857
        .dissolve(by="sensor_id", as_index=False)
        [["sensor_id", "geometry"]]
    )
    print(f"🔹 Dissolved into {len(edges_single)} sensor geometries (one per sensor)")

    # 4b) Save that “one-row per sensor” geometry for inspection
    edges_single.to_file(out_dir / "sensor_paths.gpkg",
                         layer="sensor_paths", driver="GPKG")
    print(f"🔹 Wrote matched sensor edges (dissolved) to sensor_paths.gpkg")

    # 5) Assign land-use by nearest centroid (only residential/retail/commercial)
    flat = assign_landuse_by_nearest(
        edges_single, land,
        allowed={"residential", "retail", "commercial"}
    )
    print(f"🔹 Assigned land-use for {len(flat)} sensors (via nearest-centroid logic)")

    # 6) Merge back onto edges_single so we can write final outputs
    final = edges_single.merge(flat, on="sensor_id", how="left")
    # Bring in sensor_name for CSV
    final = final.merge(
        sensors[["sensor_id", "sensor_name"]].drop_duplicates(),
        on="sensor_id", how="left"
    )

    # 7) Write out sensor_landuse.gpkg & sensor_landuse.csv
    final.to_file(out_dir / "sensor_landuse.gpkg",
                  layer="sensor_landuse", driver="GPKG")
    final[["sensor_id", "sensor_name", "landuse"]].to_csv(
        out_dir / "sensor_landuse.csv", index=False
    )
    print("✅ Done – outputs written:")
    print(f"   • {out_dir/'sensor_paths.gpkg'}")
    print(f"   • {out_dir/'sensor_landuse.gpkg'}")
    print(f"   • {out_dir/'sensor_landuse.csv'}")


if __name__ == "__main__":
    main()
