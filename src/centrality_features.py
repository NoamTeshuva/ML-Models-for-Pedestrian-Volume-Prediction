# src/centrality_features.py

import os
import osmnx as ox
import networkx as nx
import momepy
import geopandas as gpd
from datetime import datetime


def main():
    print("Downloading Melbourne street network...")
    G = ox.graph_from_place('Melbourne, Australia', network_type='walk')

    print("Saving network as shapefile...")
    ox.save_graph_shapefile(G, filepath='../data/osm/melbourne_street_network')

    print("Loading shapefile into GeoDataFrame...")
    streets = gpd.read_file('../data/osm/melbourne_street_network/edges.shp')

    print("Converting to NetworkX graph...")
    graph = momepy.gdf_to_nx(streets, approach="primal", length='length')

    print('Calculating closeness centrality at {}'.format(datetime.now()))
    edge_centrality = nx.closeness_centrality(nx.line_graph(graph))
    nx.set_edge_attributes(graph, edge_centrality, 'closeness')

    print("Saving centrality back to GeoDataFrame...")
    edges = ox.graph_to_gdfs(graph, nodes=False)
    edges.to_file('data/processed/melbourne_edges_with_centrality.shp')

    print("✅ Done!")


if __name__ == "__main__":
    main()
