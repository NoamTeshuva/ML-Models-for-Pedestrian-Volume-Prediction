#!/usr/bin/env python3
"""
centrality_features.py

This module serves two purposes:
 1) Provides compute_centrality(G, gdf) for on‑the‑fly use in the Flask API.
 2) Retains the original main() CLI workflow for batch processing and file I/O.
"""
import sys
import ast
from pathlib import Path
import pandas as pd
import geopandas as gpd
import fiona
import osmnx as ox
import networkx as nx
from osmnx.truncate import truncate_graph_polygon
from .centrality_features_fast import compute_centrality_fast


def normalize_osm(x):
    """Normalize an OSMID field into a list of ints."""
    if isinstance(x, list):
        return [int(i) for i in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return [int(i) for i in ast.literal_eval(s)]
            except Exception:
                return [int(i.strip()) for i in s[1:-1].split(",") if i.strip().isdigit()]
        elif s.isdigit():
            return [int(s)]
    try:
        return [int(x)]
    except Exception:
        return []


def compute_centrality(G, gdf, sample_size: int = None):
    """
    Compute and attach betweenness & closeness centrality to an edge GeoDataFrame.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    gdf : GeoDataFrame
        Edge GeoDataFrame; must contain column 'u' for source node.
    sample_size : int, optional
        Number of nodes to sample for approximate betweenness. Defaults to 500 or total nodes.

    Returns
    -------
    GeoDataFrame
        The same gdf with two new columns:
          - 'betweenness'
          - 'closeness'
    """
    n = len(G.nodes)
    k = sample_size if sample_size is not None else min(500, n)

    between, close = compute_centrality_fast(G, gdf, k=k)

    if 'u' not in gdf.columns:
        raise KeyError("GeoDataFrame must contain column 'u' for source node IDs.")

    gdf['betweenness'] = gdf['u'].map(between).fillna(0)
    gdf['closeness']   = gdf['u'].map(close).fillna(0)
    return gdf


def main():
    """
    Command-line interface for batch processing centrality on NYC data.
    Writes:
      - nyc_sensor_centrality.gpkg
      - sensor_centrality.csv
    """
    root = Path(__file__).resolve().parents[2]
    sensor_csv = root / "data" / "processed" / "NewYork" / "sensor_with_highway.csv"
    gpkg_file  = root / "data" / "osm" / "newyork_street_network" / "newyork_network.gpkg"
    out_dir    = root / "data" / "processed" / "NewYork"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_gpkg  = out_dir / "nyc_sensor_centrality.gpkg"
    sensor_out = out_dir / "sensor_centrality.csv"
    layer_name = "nyc_sensor_centrality"

    # Load sensor lookup
    df_s = pd.read_csv(sensor_csv, usecols=["sensor_name","osmid"], dtype=str)
    df_s["osmid"] = df_s["osmid"].str.strip()
    df_s = df_s.dropna(subset=["sensor_name","osmid"]).drop_duplicates()
    df_s["osmid_list"] = df_s["osmid"].apply(normalize_osm)
    df_s = df_s.explode("osmid_list").dropna(subset=["osmid_list"])
    df_s["osmid"] = df_s["osmid_list"].astype(int)
    all_osmids = set(df_s["osmid"])
    print(f"✅ Loaded {len(df_s)} sensor entries ({len(all_osmids)} unique OSMIDs)")

    # Load and validate network
    layers = set(fiona.listlayers(str(gpkg_file)))
    if not {"nodes","edges"} <= layers:
        sys.exit(f"❌ GeoPackage must contain 'nodes' & 'edges'; got: {layers}")
    nodes_gdf = gpd.read_file(gpkg_file, layer="nodes")
    edges_gdf = gpd.read_file(gpkg_file, layer="edges")
    print(f"✅ Network loaded: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")

    # Build graph
    if "index" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.set_index("index")
    elif "osmid" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.set_index("osmid")
    else:
        sys.exit("❌ 'nodes' must have 'index' or 'osmid'")

    edges_gdf = edges_gdf.set_index(["u","v","key"])
    G = ox.graph_from_gdfs(nodes_gdf, edges_gdf)
    G = ox.project_graph(G, to_crs="EPSG:3857")
    print("✅ Graph projected to EPSG:3857")

    # Compute centrality
    edges_sub = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    edges_c  = compute_centrality(G, edges_sub)

    # Write outputs
    edges_c.to_file(full_gpkg, layer=layer_name, driver="GPKG")
    print(f"✅ GeoPackage written: {full_gpkg.name} (layer: {layer_name})")

    # Sensor‐only centrality CSV
    rows = []
    for osmid, triples in compute_osmid_to_uvk(df_s, edges_sub).items():
        cvals = [(edges_c.loc[(edges_c.u==u)&(edges_c.v==v), 'closeness'].iat[0]
                 + edges_c.loc[(edges_c.u==u)&(edges_c.v==v), 'closeness'].iat[0]) / 2
                 for (u,v,k) in triples]
        bvals = [edges_c.loc[(edges_c.u==u)&(edges_c.v==v), 'betweenness'].iat[0]
                 for (u,v,k) in triples]
        rows.append({
            'sensor_name': df_s.loc[df_s.osmid==osmid,'sensor_name'].iat[0],
            'osmid': osmid,
            'closeness': sum(cvals)/len(cvals),
            'betweenness': sum(bvals)/len(bvals)
        })
    pd.DataFrame(rows).to_csv(sensor_out, index=False)
    print(f"✅ CSV written: {sensor_out.name}")


def compute_osmid_to_uvk(df_s, edges_sub):
    # Helper to regenerate the osmid->(u,v,key) mapping for the main() CSV routine
    mapping = df_s.groupby('osmid')[['u','v','key']].apply(lambda df: [tuple(x) for x in df.values])
    return mapping.to_dict()

if __name__ == "__main__":
    main()
