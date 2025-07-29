#!/usr/bin/env python3
"""
test_dynamic_landuse.py

Test script to verify the dynamic land use functionality works correctly.
"""
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering.landuse_features import get_landuse_polygons, cleanup_temp_files
import geopandas as gpd
import osmnx as ox

def test_place_based_landuse():
    """Test land use generation for a specific place."""
    print("🧪 Testing place-based land use generation...")
    
    # Test with Monaco
    place = "Monaco"
    landuse_gdf = get_landuse_polygons(place=place)
    
    print(f"✅ Generated land use for {place}")
    print(f"   - Number of polygons: {len(landuse_gdf)}")
    print(f"   - Land use categories: {landuse_gdf['landuse'].unique()}")
    print(f"   - CRS: {landuse_gdf.crs}")
    
    return landuse_gdf

def test_bbox_based_landuse():
    """Test land use generation for a bounding box."""
    print("\n🧪 Testing bbox-based land use generation...")
    
    # Test with a bounding box around Monaco
    bbox = (7.4, 43.7, 7.5, 43.8)  # Monaco area
    landuse_gdf = get_landuse_polygons(bbox=bbox)
    
    print(f"✅ Generated land use for bbox {bbox}")
    print(f"   - Number of polygons: {len(landuse_gdf)}")
    print(f"   - Land use categories: {landuse_gdf['landuse'].unique()}")
    print(f"   - CRS: {landuse_gdf.crs}")
    
    return landuse_gdf

def test_caching():
    """Test that caching works correctly."""
    print("\n🧪 Testing caching functionality...")
    
    place = "Monaco"
    
    # First call should download
    print("   First call (should download)...")
    landuse1 = get_landuse_polygons(place=place)
    
    # Second call should load from cache
    print("   Second call (should load from cache)...")
    landuse2 = get_landuse_polygons(place=place)
    
    # Both should be identical
    if landuse1.equals(landuse2):
        print("✅ Caching works correctly!")
    else:
        print("❌ Caching failed - results differ")
    
    return landuse1, landuse2

def test_edge_case():
    """Test edge case with no land use data."""
    print("\n🧪 Testing edge case with no land use data...")
    
    # Test with a very small area that likely has no land use data
    bbox = (0, 0, 0.001, 0.001)  # Very small area in the ocean
    landuse_gdf = get_landuse_polygons(bbox=bbox)
    
    print(f"✅ Handled edge case with bbox {bbox}")
    print(f"   - Number of polygons: {len(landuse_gdf)}")
    print(f"   - Empty GeoDataFrame: {landuse_gdf.empty}")
    
    return landuse_gdf

def test_cleanup():
    """Test cleanup functionality."""
    print("\n🧪 Testing cleanup functionality...")
    
    # Run cleanup (this will delete files older than 24 hours)
    cleanup_temp_files(max_age_hours=24)
    print("✅ Cleanup completed")

def main():
    """Run all tests."""
    print("🚀 Starting dynamic land use tests...\n")
    
    try:
        # Test place-based generation
        test_place_based_landuse()
        
        # Test bbox-based generation
        test_bbox_based_landuse()
        
        # Test caching
        test_caching()
        
        # Test edge case
        test_edge_case()
        
        # Test cleanup
        test_cleanup()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())