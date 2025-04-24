import os
import osmnx as ox

def download_street_network():
    output_dir = '../data/osm/melbourne_street_network'
    os.makedirs(output_dir, exist_ok=True)

    print(f"OSMnx version: {ox.__version__}")
    print("Downloading Melbourne street network (including walkable and vehicle roads)...")

    # ✅ Custom filter: keep both pedestrian and vehicle roads
    custom_filter = '["highway"~"footway|pedestrian|path|cycleway|steps|residential|primary|secondary|tertiary|unclassified|living_street"]'

    G = ox.graph_from_place(
        'Melbourne, Australia',
        custom_filter=custom_filter,
        simplify=True
    )

    print("Converting graph to GeoDataFrames...")
    nodes, edges = ox.graph_to_gdfs(G)

    print("Saving edges and nodes as GeoPackage (gpkg)...")
    edges.to_file(os.path.join(output_dir, 'melbourne_network.gpkg'), layer='edges', driver='GPKG')
    nodes.to_file(os.path.join(output_dir, 'melbourne_network.gpkg'), layer='nodes', driver='GPKG')

    print(f"✅ GeoPackage saved successfully in: {output_dir}/melbourne_network.gpkg")

if __name__ == "__main__":
    download_street_network()
