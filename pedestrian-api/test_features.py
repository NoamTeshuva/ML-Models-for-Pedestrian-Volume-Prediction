#!/usr/bin/env python3
"""
test_features.py

Simple unit tests for the refactored feature engineering modules.
Tests basic functionality and error handling following CLAUDE.md guidelines.
"""
import unittest
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from datetime import datetime

from feature_engineering.landuse_features import (
    validate_coordinates, validate_place_name,
    get_landuse_polygons, compute_landuse_edges
)
from feature_engineering.centrality_features import (
    normalize_osm_id, compute_centrality, CentralityError
)
from feature_engineering.highway_features import (
    normalize_highway_tag, compute_highway, HighwayError
)
from feature_engineering.feature_pipeline import (
    run_feature_pipeline, prepare_model_features, PipelineError
)

class TestLandUseFeatures(unittest.TestCase):
    """Test land use feature extraction functions."""
    
    def test_validate_coordinates(self):
        """Test coordinate validation."""
        # Valid coordinates
        self.assertTrue(validate_coordinates((-180, -90, 180, 90)))
        self.assertTrue(validate_coordinates((0, 0, 1, 1)))
        
        # Invalid coordinates
        self.assertFalse(validate_coordinates((-200, 0, 0, 0)))  # lon out of range
        self.assertFalse(validate_coordinates((0, -100, 0, 0)))  # lat out of range
        self.assertFalse(validate_coordinates((1, 0, 0, 0)))     # min >= max
    
    def test_validate_place_name(self):
        """Test place name validation."""
        # Valid names
        self.assertTrue(validate_place_name("Monaco"))
        self.assertTrue(validate_place_name("Tel Aviv, Israel"))
        
        # Invalid names
        self.assertFalse(validate_place_name(""))
        self.assertFalse(validate_place_name("   "))
        self.assertFalse(validate_place_name(None))
    
    def test_compute_landuse_edges_empty_input(self):
        """Test landuse computation with empty input."""
        empty_edges = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
        
        try:
            result = compute_landuse_edges(empty_edges)
            # Should handle empty input gracefully
            self.assertEqual(len(result), 0)
        except Exception as e:
            # Should raise appropriate error for empty input
            self.assertIn("empty", str(e).lower())

class TestCentralityFeatures(unittest.TestCase):
    """Test centrality feature extraction functions."""
    
    def test_normalize_osm_id(self):
        """Test OSMID normalization."""
        # Test various input formats
        self.assertEqual(normalize_osm_id(123), [123])
        self.assertEqual(normalize_osm_id("456"), [456])
        self.assertEqual(normalize_osm_id([789, 101]), [789, 101])
        self.assertEqual(normalize_osm_id("[112, 131]"), [112, 131])
        self.assertEqual(normalize_osm_id(""), [])
        self.assertEqual(normalize_osm_id(None), [])
    
    def test_compute_centrality_simple_graph(self):
        """Test centrality computation on simple graph."""
        # Create simple test graph
        G = nx.path_graph(4)
        nx.set_edge_attributes(G, 1.0, 'length')
        
        # Create test edges GeoDataFrame
        edges_gdf = gpd.GeoDataFrame({
            'u': [0, 1, 2],
            'v': [1, 2, 3],
            'geometry': [LineString([(0, 0), (1, 0)]) for _ in range(3)]
        }, crs="EPSG:4326")
        
        # Compute centrality
        result = compute_centrality(G, edges_gdf)
        
        # Check result structure
        self.assertIn('betweenness', result.columns)
        self.assertIn('closeness', result.columns)
        self.assertEqual(len(result), 3)
    
    def test_compute_centrality_invalid_input(self):
        """Test centrality computation with invalid input."""
        with self.assertRaises(CentralityError):
            compute_centrality("not_a_graph", gpd.GeoDataFrame())

class TestHighwayFeatures(unittest.TestCase):
    """Test highway feature extraction functions."""
    
    def test_normalize_highway_tag(self):
        """Test highway tag normalization."""
        # Test various input formats
        self.assertEqual(normalize_highway_tag("primary"), "primary")
        self.assertEqual(normalize_highway_tag(["secondary", "tertiary"]), "secondary")
        self.assertEqual(normalize_highway_tag("residential;service"), "residential")
        self.assertEqual(normalize_highway_tag(None), "unclassified")
        self.assertEqual(normalize_highway_tag(""), "unclassified")
    
    def test_compute_highway_basic(self):
        """Test basic highway computation."""
        # Create test edges
        edges_gdf = gpd.GeoDataFrame({
            'osmid': ['123', '456'],
            'highway': ['primary', 'secondary'],
            'geometry': [LineString([(0, 0), (1, 0)]) for _ in range(2)]
        }, crs="EPSG:4326")
        
        result = compute_highway(edges_gdf)
        
        # Check result structure
        self.assertIn('highway', result.columns)
        self.assertEqual(len(result), 2)
        self.assertIn(result.iloc[0]['highway'], ['primary', 'unclassified'])

class TestFeaturePipeline(unittest.TestCase):
    """Test the unified feature pipeline."""
    
    def test_prepare_model_features(self):
        """Test model feature preparation."""
        # Create test GeoDataFrame with required features
        test_features = gpd.GeoDataFrame({
            'length': [100.0, 200.0],
            'betweenness': [0.1, 0.2],
            'closeness': [0.3, 0.4],
            'Hour': [10, 14],
            'is_weekend': [0, 1],
            'time_of_day': ['morning', 'afternoon'],
            'land_use': ['residential', 'commercial'],
            'highway': ['primary', 'secondary'],
            'geometry': [LineString([(0, 0), (1, 0)]) for _ in range(2)]
        }, crs="EPSG:4326")
        
        result = prepare_model_features(test_features)
        
        # Check result structure
        self.assertEqual(len(result.columns), 8)  # Should have all required features
        self.assertEqual(len(result), 2)
        
        # Check column order matches expected
        expected_cols = [
            "length", "betweenness", "closeness", "Hour", 
            "is_weekend", "time_of_day", "land_use", "highway"
        ]
        self.assertEqual(list(result.columns), expected_cols)
    
    def test_pipeline_validation(self):
        """Test pipeline input validation."""
        with self.assertRaises(PipelineError):
            # Should fail with no place or bbox
            run_feature_pipeline()

class TestIntegration(unittest.TestCase):
    """Integration tests for combined functionality."""
    
    def test_small_area_pipeline(self):
        """Test pipeline on a very small area (may use mock data in real implementation)."""
        # This test would ideally use a small, reliable area or mock data
        # For demonstration, we'll test the validation only
        
        try:
            # This might fail due to network issues, but should validate inputs
            features_gdf, metadata = run_feature_pipeline(
                place="Monaco",
                timestamp="2024-01-15T12:00:00"
            )
            
            # If successful, validate structure
            self.assertIsInstance(features_gdf, gpd.GeoDataFrame)
            self.assertIsInstance(metadata, dict)
            self.assertIn('processing_time', metadata)
            self.assertIn('network_stats', metadata)
            
        except PipelineError as e:
            # Expected for network/data issues - check it's a reasonable error
            self.assertIn('error', e.to_dict())
            self.assertIn('code', e.to_dict())
        except Exception as e:
            # Unexpected errors should be investigated
            self.fail(f"Pipeline failed with unexpected error: {e}")

def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_classes = [
        TestLandUseFeatures,
        TestCentralityFeatures, 
        TestHighwayFeatures,
        TestFeaturePipeline,
        TestIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    print("Running feature engineering tests...")
    result = run_tests()
    
    if result.wasSuccessful():
        print(f"\nAll {result.testsRun} tests passed successfully!")
    else:
        print(f"\n{len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        
        if result.failures:
            print("\nFailures:")
            for test, failure in result.failures:
                print(f"- {test}: {failure}")
        
        if result.errors:
            print("\nErrors:")
            for test, error in result.errors:
                print(f"- {test}: {error}")