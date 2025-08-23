#!/usr/bin/env python3
"""
One-time script to build 10x10 km tiles covering Israel with OSM walk network data.
Saves to tiles/data/{tile_id}.parquet and creates tiles/index.geojson.
"""

import os
import time
import hashlib
import logging
from typing import List, Tuple
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import Polygon, LineString
import pyproj
from pathlib import Path

# Configuration
TILE_SIZE_KM = 10  # 10x10 km tiles
ISRAEL_BOUNDS = (34.2, 29.4, 35.9, 33.4)  # west, south, east, north
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure OSMnx
ox.config(use_cache=True, overpass_rate_limit=True, timeout=180)


def create_edge_id(osmid: str, geometry: LineString) -> str:
    """Create stable edge ID from osmid and geometry hash."""
    geom_wkb = geometry.wkb
    geom_hash = hashlib.md5(geom_wkb).hexdigest()[:16]
    return f"{osmid}_{geom_hash}"


def generate_tiles(bounds: Tuple[float, float, float, float], tile_size_km: int) -> List[dict]:
    """Generate 10x10 km tile grid covering Israel bounds."""
    west, south, east, north = bounds
    
    # Convert to UTM for metric calculations (Israel is in UTM Zone 36N)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    transformer_back = pyproj.Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)
    
    # Convert bounds to UTM
    west_utm, south_utm = transformer.transform(west, south)
    east_utm, north_utm = transformer.transform(east, north)
    
    tile_size_m = tile_size_km * 1000
    tiles = []
    
    x = west_utm
    tile_x = 0
    while x < east_utm:
        y = south_utm
        tile_y = 0
        while y < north_utm:
            # Create tile bounds in UTM
            tile_west_utm = x
            tile_south_utm = y
            tile_east_utm = min(x + tile_size_m, east_utm)
            tile_north_utm = min(y + tile_size_m, north_utm)
            
            # Convert back to WGS84
            tile_west, tile_south = transformer_back.transform(tile_west_utm, tile_south_utm)
            tile_east, tile_north = transformer_back.transform(tile_east_utm, tile_north_utm)
            
            tile_id = f"tile_{tile_x}_{tile_y}"
            
            # Create polygon for this tile
            poly = Polygon([
                (tile_west, tile_south),
                (tile_east, tile_south),
                (tile_east, tile_north),
                (tile_west, tile_north),
                (tile_west, tile_south)
            ])
            
            tiles.append({
                'tile_id': tile_id,
                'geometry': poly,
                'bounds': (tile_west, tile_south, tile_east, tile_north)
            })
            
            y += tile_size_m
            tile_y += 1
        x += tile_size_m
        tile_x += 1
    
    return tiles


def download_tile_network(tile_bounds: Tuple[float, float, float, float], tile_id: str) -> gpd.GeoDataFrame:
    """Download walk network for a single tile with retry logic."""
    west, south, east, north = tile_bounds
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            logger.info(f"Downloading {tile_id} (attempt {attempt + 1}/{RETRY_ATTEMPTS})")
            
            # Download walk network within bounding box
            G = ox.graph_from_bbox(
                north, south, east, west,
                network_type='walk',
                simplify=True,
                retain_all=False
            )
            
            if len(G.edges) == 0:
                logger.warning(f"No edges found for {tile_id}")
                return gpd.GeoDataFrame(columns=['edge_id', 'osmid', 'highway', 'geometry'], crs='EPSG:4326')
            
            # Convert to GeoDataFrame
            edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
            
            # Clean and prepare data
            edges_gdf = edges_gdf.reset_index()
            
            # Ensure geometry is LineString and in EPSG:4326
            edges_gdf = edges_gdf.to_crs('EPSG:4326')
            edges_gdf = edges_gdf[edges_gdf.geometry.type == 'LineString'].copy()
            
            # Create stable edge IDs
            edges_gdf['osmid_str'] = edges_gdf['osmid'].astype(str)
            edges_gdf['edge_id'] = edges_gdf.apply(
                lambda row: create_edge_id(row['osmid_str'], row.geometry), axis=1
            )
            
            # Keep only required columns
            result = edges_gdf[['edge_id', 'osmid_str', 'highway', 'geometry']].copy()
            result = result.rename(columns={'osmid_str': 'osmid'})
            
            # Fill missing highway values
            result['highway'] = result['highway'].fillna('unclassified')
            
            logger.info(f"Downloaded {len(result)} edges for {tile_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error downloading {tile_id} (attempt {attempt + 1}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to download {tile_id} after {RETRY_ATTEMPTS} attempts")
                return gpd.GeoDataFrame(columns=['edge_id', 'osmid', 'highway', 'geometry'], crs='EPSG:4326')


def deduplicate_edges(all_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove duplicate edges across tiles based on edge_id."""
    logger.info(f"Deduplicating {len(all_edges)} total edges")
    
    # Remove duplicates by edge_id (keeping first occurrence)
    deduped = all_edges.drop_duplicates(subset=['edge_id'], keep='first')
    
    logger.info(f"After deduplication: {len(deduped)} unique edges")
    return deduped


def main():
    """Main function to build Israel tiles."""
    logger.info("Starting Israel tile building process")
    
    # Create output directories
    tiles_dir = Path("tiles")
    data_dir = tiles_dir / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate tile grid
    tiles = generate_tiles(ISRAEL_BOUNDS, TILE_SIZE_KM)
    logger.info(f"Generated {len(tiles)} tiles covering Israel")
    
    # Download network for each tile
    all_edges = []
    successful_tiles = []
    
    for i, tile in enumerate(tiles):
        tile_id = tile['tile_id']
        bounds = tile['bounds']
        
        logger.info(f"Processing tile {i+1}/{len(tiles)}: {tile_id}")
        
        edges_gdf = download_tile_network(bounds, tile_id)
        
        if len(edges_gdf) > 0:
            # Save tile to parquet
            output_path = data_dir / f"{tile_id}.parquet"
            edges_gdf.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(edges_gdf)} edges to {output_path}")
            
            all_edges.append(edges_gdf)
            successful_tiles.append(tile)
        else:
            logger.warning(f"Skipping empty tile {tile_id}")
    
    if not all_edges:
        logger.error("No tiles were successfully processed!")
        return
    
    # Create tiles index
    tiles_gdf = gpd.GeoDataFrame(successful_tiles, crs='EPSG:4326')
    index_path = tiles_dir / "index.geojson"
    tiles_gdf.to_file(index_path, driver='GeoJSON')
    logger.info(f"Saved tile index to {index_path}")
    
    # Deduplicate edges across all tiles
    all_edges_combined = pd.concat(all_edges, ignore_index=True)
    all_edges_gdf = gpd.GeoDataFrame(all_edges_combined, crs='EPSG:4326')
    deduped_edges = deduplicate_edges(all_edges_gdf)
    
    logger.info(f"Tile building complete!")
    logger.info(f"- Processed {len(successful_tiles)} successful tiles")
    logger.info(f"- Total unique edges: {len(deduped_edges)}")
    logger.info(f"- Tiles saved to: {data_dir}")
    logger.info(f"- Index saved to: {index_path}")


if __name__ == "__main__":
    main()