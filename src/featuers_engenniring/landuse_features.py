"""
landuse_features.py

Compute OSM 'landuse' for each street segment.

Inputs:
  - data/osm/melbourne_street_network/melbourne_network.gpkg (layer 'edges')
Outputs:
  - data/processed/melbourne_landuse_features.gpkg (layer 'edges' with new 'landuse' field)
"""

import os
import geopandas as gpd
import osmnx as ox

def most_freq(s):
    """Return the mode of a Series, or None if empty."""
    m = s.mode()
    return m.iloc[0] if not m.empty else None

def main():
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    GPkg = os.path.join(BASE, "data", "osm", "melbourne_street_network",
                        "melbourne_network.gpkg")
    OUT  = os.path.join(BASE, "data", "processed",
                        "melbourne_landuse_features.gpkg")

    # 1) load street segments and reproject for buffering
    print("📥 Loading street segments…")
    roads = (
        gpd.read_file(GPkg, layer="edges")
           .to_crs(epsg=3857)
    )

    # 2) download all OSM landuse polygons
    print("🌐 Downloading landuse polygons from OSM…")
    tags = {"landuse": True}
    lu = (
        ox.features_from_place("Melbourne, Australia", tags)
          .to_crs(epsg=3857)
    )
    # keep only actual polygons and rename the tag column
    lu = (
        lu[lu.geometry.type.isin(["Polygon", "MultiPolygon"])]
          .rename(columns={"landuse": "landuse_tag"})
          [["landuse_tag", "geometry"]]
    )

    # 3a) assign by direct intersection
    print("🔗 Assigning landuse by direct intersection…")
    intersect = gpd.sjoin(
        roads, lu,
        how="left",
        predicate="intersects",
    )
    # collapse multiple matches: take first landuse_tag per road index
    direct = intersect.groupby(intersect.index)["landuse_tag"].first()
    roads["landuse"] = roads.index.map(direct)

    # 3b) for roads still missing, buffer & pick the most frequent landuse_tag
    print("🧹 Buffering segments missing landuse and picking mode…")
    tofix = roads[roads["landuse"].isna()].copy()
    tofix["geometry"] = tofix.geometry.buffer(50)  # 50m

    buf_join = gpd.sjoin(
        tofix, lu,
        how="left",
        predicate="intersects",
    )
    buffered = buf_join.groupby(buf_join.index)["landuse_tag"].agg(most_freq)
    roads.loc[buffered.index, "landuse"] = buffered

    # 4) save out
    print(f"💾 Saving edges with landuse to {OUT} …")
    roads.to_file(OUT, layer="edges", driver="GPKG")

    print("✅ Done.")

if __name__ == "__main__":
    main()
