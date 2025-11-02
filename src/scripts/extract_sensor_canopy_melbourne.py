#!/usr/bin/env python3
"""
Extract canopy features for real Melbourne sensors only.
Uses the v4 sampling approach but focuses on sensor-specific edges.
"""
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features, windows
from shapely.geometry import Point
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sensor_edges_with_geometry(sensor_edge_csv, network_gpkg):
    """Get sensor edges with their geometries from the network."""
    print("Loading sensor-edge mapping...")
    sensor_edges = pd.read_csv(sensor_edge_csv)
    
    print("Loading network edges...")
    edges_gdf = gpd.read_file(network_gpkg, layer='edges')
    
    print(f"Sensor edges: {len(sensor_edges)} mappings")
    print(f"Network edges: {len(edges_gdf)} total edges")
    
    # Get unique sensor names and edge IDs
    unique_sensors = sensor_edges['sensor_id'].unique()
    unique_edge_ids = sensor_edges['edge_osmid'].unique()
    
    print(f"Unique sensors: {len(unique_sensors)}")
    print(f"Unique edge IDs: {len(unique_edge_ids)}")
    
    # Convert edge IDs to strings for matching
    unique_edge_ids_str = [str(x) for x in unique_edge_ids]
    
    # Filter network to only edges used by sensors
    sensor_network = edges_gdf[edges_gdf['osmid'].isin(unique_edge_ids_str)].copy()
    
    print(f"Filtered network: {len(sensor_network)} edges")
    
    # Merge with sensor mapping (convert edge_osmid to string for matching)
    sensor_edges['edge_osmid_str'] = sensor_edges['edge_osmid'].astype(str)
    result = sensor_edges.merge(
        sensor_network[['osmid', 'geometry']], 
        left_on='edge_osmid_str', 
        right_on='osmid', 
        how='left'
    )
    
    # Convert to GeoDataFrame
    result_gdf = gpd.GeoDataFrame(result, geometry='geometry', crs=sensor_network.crs)
    
    print(f"Final sensor edges with geometry: {len(result_gdf)}")
    return result_gdf

def sample_ndvi_for_sensor_edges(sensor_edges_gdf, raster_path, buffer_m=50, ndvi_threshold=0.3):
    """Sample NDVI values for buffered sensor edges."""
    results = []
    
    print(f"Opening NDVI raster: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        
        print(f"Raster CRS: {raster_crs}")
        print(f"Raster bounds: {raster_bounds}")
        
        # Transform sensor edges to raster CRS
        print("Transforming sensor edges to raster CRS...")
        sensor_edges_proj = sensor_edges_gdf.to_crs(raster_crs)
        
        # Create buffers
        print(f"Creating {buffer_m}m buffers...")
        sensor_edges_proj['buffer_geom'] = sensor_edges_proj.geometry.buffer(buffer_m)
        
        total_edges = len(sensor_edges_proj)
        print(f"Processing {total_edges} sensor edges...")
        
        for idx, row in sensor_edges_proj.iterrows():
            if idx % 50 == 0:
                print(f"  Processing edge {idx+1}/{total_edges}")
            
            sensor_id = row['sensor_id']
            edge_osmid = row['edge_osmid']
            buffer_geom = row['buffer_geom']
            
            try:
                # Get bounding box for windowed reading
                bbox = buffer_geom.bounds
                
                # Convert to raster window
                window = windows.from_bounds(*bbox, src.transform)
                window = window.intersection(windows.Window(0, 0, src.width, src.height))
                
                if window.width <= 0 or window.height <= 0:
                    print(f"    Skipping edge {edge_osmid} (outside raster bounds)")
                    continue
                
                # Read windowed data
                window_data = src.read(1, window=window)
                window_transform = windows.transform(window, src.transform)
                
                # Create mask for the buffer
                mask = features.rasterize(
                    [buffer_geom],
                    out_shape=window_data.shape,
                    transform=window_transform,
                    fill=0,
                    default_value=1,
                    dtype=np.uint8
                )
                
                # Apply mask to get NDVI values in buffer
                mask_pixels = mask == 1
                valid_data = (window_data != src.nodata) & (window_data > -1) & (window_data < 1)
                valid_mask = mask_pixels & valid_data
                
                if not valid_mask.any():
                    print(f"    No valid NDVI data for edge {edge_osmid}")
                    continue
                
                ndvi_values = window_data[valid_mask]
                n_valid = len(ndvi_values)
                
                # Calculate metrics
                ndvi_mean = float(np.mean(ndvi_values))
                canopy_pixels = np.sum(ndvi_values >= ndvi_threshold)
                canopy_pct = float(canopy_pixels / n_valid) if n_valid > 0 else 0.0
                
                results.append({
                    'sensor_id': sensor_id,
                    'edge_osmid': edge_osmid,
                    'sensor_canopy_pct': canopy_pct,
                    'sensor_ndvi_mean': ndvi_mean,
                    'sensor_canopy_valid_frac': 1.0,  # All sampled pixels were valid
                    'n_edges': 1,  # Each row is one edge
                    'n_pixels_sampled': n_valid
                })
                
            except Exception as e:
                print(f"    Error processing edge {edge_osmid}: {e}")
                continue
    
    return pd.DataFrame(results)

def aggregate_by_sensor(edge_results_df):
    """Aggregate edge-level results to sensor level."""
    print("Aggregating results by sensor...")
    
    sensor_results = edge_results_df.groupby('sensor_id').agg({
        'sensor_canopy_pct': 'mean',
        'sensor_ndvi_mean': 'mean', 
        'sensor_canopy_valid_frac': 'mean',
        'n_edges': 'count',
        'n_pixels_sampled': 'sum'
    }).reset_index()
    
    print(f"Generated results for {len(sensor_results)} sensors")
    return sensor_results

def main():
    # File paths
    sensor_edge_csv = "data/processed/melbourne/csv/sensor_edge_map_2019_melbourne.csv"
    network_gpkg = "data/osm/melbourne_street_network/melbourne_network.gpkg"
    raster_path = "data/external/melbourne/green.tif"
    output_csv = "data/processed/melbourne/csv/sensor_canopy_features_REAL_melbourne.csv"
    
    print("=== Melbourne Real Sensor Canopy Extraction ===")
    print(f"Sensor edges: {sensor_edge_csv}")
    print(f"Network: {network_gpkg}")
    print(f"NDVI raster: {raster_path}")
    print(f"Output: {output_csv}")
    print()
    
    try:
        # 1. Get sensor edges with geometry
        sensor_edges_gdf = get_sensor_edges_with_geometry(sensor_edge_csv, network_gpkg)
        
        # 2. Sample NDVI for each edge
        edge_results = sample_ndvi_for_sensor_edges(sensor_edges_gdf, raster_path)
        
        if len(edge_results) == 0:
            print("ERROR: No results generated!")
            return 1
        
        # 3. Aggregate by sensor
        sensor_results = aggregate_by_sensor(edge_results)
        
        # 4. Save results
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        sensor_results.to_csv(output_csv, index=False)
        
        print(f"\n=== RESULTS ===")
        print(f"Processed {len(edge_results)} edges")
        print(f"Generated features for {len(sensor_results)} sensors")
        print(f"Output saved: {output_csv}")
        
        # Show sample results
        print(f"\nSample sensor results:")
        print(sensor_results.head())
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())