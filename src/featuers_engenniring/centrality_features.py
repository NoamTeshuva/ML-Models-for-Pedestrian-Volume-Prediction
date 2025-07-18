#!/usr/bin/env python3
"""
centrality_features_ny.py

Compute closeness & betweenness centrality for New York pedestrian sensors using a 200m buffer.

Outputs:
- nyc_sensor_centrality.gpkg  (sensor edges with centrality attributes)
- sensor_centrality.csv       (tabular format)
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


def normalize_osm(x):
    """Normalize OSMID field into a list of ints."""
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


def main():
    root = Path(__file__).resolve().parents[2]
    sensor_csv = root / "data" / "processed" / "NewYork" / "sensor_with_highway.csv"
    gpkg_file = root / "data" / "osm" / "newyork_street_network" / "newyork_network.gpkg"
    out_dir = root / "data" / "processed" / "NewYork"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_gpkg = out_dir / "nyc_sensor_centrality.gpkg"
    sensor_out = out_dir / "sensor_centrality.csv"
    layer_name = "nyc_sensor_centrality"

    df_s = pd.read_csv(sensor_csv, usecols=["sensor_name", "osmid"], dtype=str)
    df_s["osmid"] = df_s["osmid"].str.strip()
    df_s = df_s.dropna(subset=["sensor_name", "osmid"]).drop_duplicates()
    df_s["osmid_list"] = df_s["osmid"].apply(normalize_osm)
    df_s = df_s.explode("osmid_list").dropna(subset=["osmid_list"])
    df_s["osmid"] = df_s["osmid_list"].astype(int)
    all_osmids = set(df_s["osmid"])
    print(f"✅ Loaded {len(df_s)} sensor entries ({len(all_osmids)} unique OSMIDs)")

    # Load network
    layers = set(fiona.listlayers(str(gpkg_file)))
    if not {"nodes", "edges"} <= layers:
        sys.exit(f"❌ GeoPackage must contain 'nodes' & 'edges'; got: {layers}")
    nodes_gdf = gpd.read_file(gpkg_file, layer="nodes")
    edges_gdf = gpd.read_file(gpkg_file, layer="edges")
    print(f"✅ Network loaded: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")

    if "index" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.set_index("index")
    elif "osmid" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.set_index("osmid")
    else:
        sys.exit("❌ 'nodes' must have 'index' or 'osmid'")

    edges_gdf = edges_gdf.set_index(["u", "v", "key"])
    G = ox.graph_from_gdfs(nodes_gdf, edges_gdf)
    G = ox.project_graph(G, to_crs="EPSG:3857")
    print("✅ Graph projected to EPSG:3857")

    edges_reset = edges_gdf.reset_index()[["u", "v", "key", "osmid", "geometry"]].copy()
    edges_reset["osmid_list"] = edges_reset["osmid"].apply(normalize_osm)
    edges_exp = edges_reset.explode("osmid_list").copy()
    edges_exp = edges_exp.rename(columns={"osmid_list": "osmid"})
    edges_exp["osmid"] = pd.to_numeric(edges_exp["osmid"], errors="coerce")
    edges_exp = edges_exp.dropna(subset=["osmid"])
    edges_exp["osmid"] = edges_exp["osmid"].astype(int)
    print(f"✅ Exploded to {len(edges_exp)} edge–OSMID rows.")

    matched = edges_exp[edges_exp["osmid"].isin(all_osmids)].merge(df_s, on="osmid", how="left")
    print(f"✅ Matched {len(matched)} sensor→edge rows")
    osmid_to_uvk = (
        matched.groupby("osmid")[["u", "v", "key"]]
        .apply(lambda df: [tuple(x) for x in df.values])
        .to_dict()
    )
    matched_osmids = set(osmid_to_uvk)
    unmatched = all_osmids - matched_osmids
    print(f"✅ {len(unmatched)} sensor OSMIDs did not match any edge")

    nodes_proj = ox.graph_to_gdfs(G, nodes=True, edges=False)
    sensor_nodes = {n for triples in osmid_to_uvk.values() for (u, v, k) in triples for n in (u, v)}
    pts = nodes_proj.loc[list(sensor_nodes)]
    buffer_poly = pts.geometry.unary_union.buffer(200)
    print(f"✅ Buffered {len(sensor_nodes)} nodes (200m radius)")

    G_sub = truncate_graph_polygon(G, buffer_poly)
    clos = nx.closeness_centrality(G_sub, distance="length")
    betw = nx.edge_betweenness_centrality(G_sub, weight="length")
    print("✅ Centrality computed on subgraph")

    edges_sub = ox.graph_to_gdfs(G_sub, nodes=False, edges=True).reset_index()
    edges_sub["closeness"] = (edges_sub["u"].map(clos) + edges_sub["v"].map(clos)) / 2
    edges_sub["betweenness"] = edges_sub.apply(
        lambda r: betw.get((r["u"], r["v"], r["key"]), 0), axis=1
    )

    sensor_uvk = {(u, v, k) for triples in osmid_to_uvk.values() for (u, v, k) in triples}
    sensor_edges = edges_sub[edges_sub.apply(
        lambda r: (r["u"], r["v"], r["key"]) in sensor_uvk, axis=1
    )]
    sensor_edges = sensor_edges.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:3857")

    sensor_edges.to_file(full_gpkg, layer=layer_name, driver="GPKG")
    print(f"✅ GeoPackage written: {full_gpkg.name} (layer: {layer_name})")

    rows = []
    for _, row in df_s[df_s["osmid"].isin(matched_osmids)].iterrows():
        uvk_list = osmid_to_uvk[row.osmid]
        cvals = [(clos.get(u, 0) + clos.get(v, 0)) / 2 for (u, v, k) in uvk_list]
        bvals = [betw.get((u, v, k), 0) for (u, v, k) in uvk_list]
        rows.append({
            "sensor_name": row.sensor_name,
            "osmid": row.osmid,
            "closeness": sum(cvals) / len(cvals),
            "betweenness": sum(bvals) / len(bvals),
        })
    pd.DataFrame(rows).to_csv(sensor_out, index=False)
    print(f"✅ CSV written: {sensor_out.name}")


if __name__ == "__main__":
    main()
