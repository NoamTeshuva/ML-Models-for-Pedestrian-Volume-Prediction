import os
import osmnx as ox
import networkx as nx
import geopandas as gpd
import random

def main():
    print("✅ Loading Melbourne network from GeoPackage…")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gpkg = os.path.join(BASE_DIR, "data", "osm", "melbourne_street_network", "melbourne_network.gpkg")
    edges = gpd.read_file(gpkg, layer="edges")
    nodes = gpd.read_file(gpkg, layer="nodes")

    # --- index fix ---------------------------------------------------------
    if edges.index.names != ["u", "v", "key"]:
        edges = edges.set_index(["u", "v", "key"])
    if nodes.index.name != "osmid":
        nodes = nodes.set_index("osmid")

    # remove duplicates just in case
    edges = edges[~edges.index.duplicated(keep="first")]
    nodes = nodes[~nodes.index.duplicated(keep="first")]

    # --- SAMPLE 10 RANDOM EDGES --------------------------------------------
    random.seed(42)
    sample_idx = random.sample(list(edges.index), 10)
    edges = edges.loc[sample_idx]
    # figure out which nodes are touched by those 10 edges
    us = [u for u, v, k in sample_idx]
    vs = [v for u, v, k in sample_idx]
    keep_nodes = set(us) | set(vs)
    nodes = nodes.loc[nodes.index.isin(keep_nodes)]

    # ----------------------------------------------------------------------
    print("🔄 Building graph on just 10 edges…")
    G = ox.graph_from_gdfs(nodes, edges)

    # --- centrality on small graph ----------------------------------------
    print("⚡ Exact closeness (no sampling)…")
    closeness = nx.closeness_centrality(G, distance="length")
    nx.set_node_attributes(G, closeness, "closeness")

    print("⚡ Exact betweenness (no sampling)…")
    betweenness = nx.betweenness_centrality(G, weight="length", normalized=True)
    nx.set_node_attributes(G, betweenness, "betweenness")

    # write node values back to edges
    for u, v, k_, data in G.edges(keys=True, data=True):
        data["closeness"]   = (G.nodes[u]["closeness"] + G.nodes[v]["closeness"]) / 2
        data["betweenness"] = (G.nodes[u]["betweenness"] + G.nodes[v]["betweenness"]) / 2

    print("💾 Saving small-sample results…")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    out_gpkg = os.path.join(BASE_DIR, "data", "osm", "melbourne_street_network",
                            "melbourne_network_10edges_centrality.gpkg")
    edges_gdf.to_file(out_gpkg, layer="edges", driver="GPKG")
    nodes_gdf.to_file(out_gpkg, layer="nodes", driver="GPKG")

    print("✅ Done →", out_gpkg)

if __name__ == "__main__":
    main()
