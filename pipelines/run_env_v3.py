#!/usr/bin/env python3
"""
v3 Environmental Feature Extraction Pipeline

Orchestrates topography v3 and green canopy v3 feature extraction for a city/year.
Reads config/cities.yaml to determine extractor version and parameters.
Generates coverage reports and quality metrics.

Usage:
    python pipelines/run_env_v3.py melbourne 2023
    python pipelines/run_env_v3.py --city melbourne --year 2023 --force-v3
    python pipelines/run_env_v3.py --bbox "-37.9,144.8,-37.7,145.1" --year 2023 --name melbourne_test
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import yaml

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from feature_engineering.topography_features_v3 import extract_topography_features_v3
from feature_engineering.green_canopy_features_v3 import extract_green_canopy_features_v3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_city_config(config_path: str = "config/cities.yaml") -> Dict[str, Any]:
    """Load city configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration for {len(config)} cities")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return {}


def get_city_parameters(config: Dict[str, Any], city: str) -> Dict[str, Any]:
    """Extract parameters for a specific city."""
    city_config = config.get(city, {})
    
    defaults = {
        'extractor_version': 'v2',
        'fallback_to_v2': True,
        'topography_v3': {
            'max_seg_m': 20,
            'smooth_window': 3,
            'min_valid_fraction': 0.2,
            'dem_nodata': -32768
        },
        'green_canopy_v3': {
            'buffer_m': 25,
            'ndvi_threshold': 0.2,
            'ndvi_rescale': 'auto',
            'min_coverage': 0.3,
            'treat_zero_as_nodata': True
        }
    }
    
    # Merge city config with defaults
    params = defaults.copy()
    if city_config:
        params.update(city_config)
        if 'topography_v3' in city_config:
            params['topography_v3'].update(city_config['topography_v3'])
        if 'green_canopy_v3' in city_config:
            params['green_canopy_v3'].update(city_config['green_canopy_v3'])
    
    return params


def ensure_directories(city: str) -> Tuple[Path, Path]:
    """Ensure output directories exist and return paths."""
    processed_dir = Path(f"data/processed/{city}/csv")
    reports_dir = Path("reports/metrics")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    return processed_dir, reports_dir


def compute_coverage_stats(df: pd.DataFrame, feature_columns: list) -> Dict[str, Any]:
    """Compute coverage statistics for feature columns."""
    if df.empty:
        return {
            'total_edges': 0,
            'coverage_fraction': 0.0,
            'non_null_count': 0,
            'mean_valid_fraction': 0.0
        }
    
    total_edges = len(df)
    non_null_counts = {}
    
    for col in feature_columns:
        if col in df.columns:
            non_null_counts[col] = df[col].notna().sum()
    
    if not non_null_counts:
        return {
            'total_edges': total_edges,
            'coverage_fraction': 0.0,
            'non_null_count': 0,
            'mean_valid_fraction': 0.0
        }
    
    # Use primary feature column for overall coverage
    primary_col = feature_columns[0] if feature_columns else None
    coverage_fraction = (non_null_counts.get(primary_col, 0) / total_edges 
                        if total_edges > 0 else 0.0)
    
    # Compute mean valid fraction if available
    mean_valid_fraction = 0.0
    if 'valid_seg_frac_v3' in df.columns:
        mean_valid_fraction = df['valid_seg_frac_v3'].mean()
    
    return {
        'total_edges': total_edges,
        'coverage_fraction': coverage_fraction,
        'non_null_count': non_null_counts.get(primary_col, 0),
        'mean_valid_fraction': mean_valid_fraction,
        'column_coverage': non_null_counts
    }


def run_topography_v3(network_path: str, dem_path: str, output_path: str, 
                     params: Dict[str, Any], emit_v2_names: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """Run topography v3 extraction and return success status and coverage stats."""
    logger.info("Starting topography v3 extraction...")
    
    try:
        topo_params = params['topography_v3']
        
        # Run extraction
        df = extract_topography_features_v3(
            network_path=network_path,
            dem_path=dem_path,
            output_path=output_path,
            max_seg_m=topo_params['max_seg_m'],
            smooth_window=topo_params['smooth_window'],
            min_valid_fraction=topo_params['min_valid_fraction'],
            dem_nodata=topo_params['dem_nodata'],
            emit_v2_names=emit_v2_names
        )
        
        # Compute coverage statistics
        v3_columns = ['grade_mean_pct_v3', 'grade_p85_pct_v3', 'grade_max_pct_v3', 
                      'elev_start_m_v3', 'elev_end_m_v3', 'elev_range_m_v3']
        coverage = compute_coverage_stats(df, v3_columns)
        
        logger.info(f"Topography v3 completed: {coverage['coverage_fraction']:.1%} coverage "
                   f"({coverage['non_null_count']}/{coverage['total_edges']} edges)")
        
        return True, coverage
        
    except Exception as e:
        logger.error(f"Topography v3 extraction failed: {e}")
        return False, {'total_edges': 0, 'coverage_fraction': 0.0, 'error': str(e)}


def run_green_canopy_v3(network_path: str, ndvi_path: str, output_path: str,
                       params: Dict[str, Any], emit_v2_names: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """Run green canopy v3 extraction and return success status and coverage stats."""
    logger.info("Starting green canopy v3 extraction...")
    
    try:
        canopy_params = params['green_canopy_v3']
        
        # Run extraction
        df = extract_green_canopy_features_v3(
            network_path=network_path,
            ndvi_path=ndvi_path,
            output_path=output_path,
            buffer_m=canopy_params['buffer_m'],
            ndvi_threshold=canopy_params['ndvi_threshold'],
            ndvi_rescale=canopy_params['ndvi_rescale'],
            min_coverage=canopy_params['min_coverage'],
            treat_zero_as_nodata=canopy_params['treat_zero_as_nodata'],
            emit_v2_names=emit_v2_names
        )
        
        # Compute coverage statistics
        v3_columns = ['canopy_pct_overall_v3', 'canopy_pct_spring_v3', 
                     'canopy_pct_summer_v3', 'canopy_pct_evergreen_proxy_v3']
        coverage = compute_coverage_stats(df, v3_columns)
        
        # Check for vector fallback usage
        vector_fallback_used = 0
        if 'canopy_pct_vector_v3' in df.columns:
            vector_fallback_used = df['canopy_pct_vector_v3'].notna().sum()
        
        coverage['vector_fallback_used'] = vector_fallback_used
        coverage['seasonal_bands_available'] = 'canopy_pct_spring_v3' in df.columns
        
        logger.info(f"Green canopy v3 completed: {coverage['coverage_fraction']:.1%} coverage "
                   f"({coverage['non_null_count']}/{coverage['total_edges']} edges)")
        
        if vector_fallback_used > 0:
            logger.info(f"Vector fallback used for {vector_fallback_used} edges")
        
        return True, coverage
        
    except Exception as e:
        logger.error(f"Green canopy v3 extraction failed: {e}")
        return False, {'total_edges': 0, 'coverage_fraction': 0.0, 'error': str(e)}


def should_use_v3(params: Dict[str, Any], force_v3: bool = False, force_v2: bool = False) -> bool:
    """Determine whether to use v3 or v2 extraction."""
    if force_v3:
        return True
    if force_v2:
        return False
    
    return params.get('extractor_version', 'v2') == 'v3'


def generate_coverage_report(city: str, year: int, topography_coverage: Dict[str, Any],
                           canopy_coverage: Dict[str, Any], processing_time: float,
                           extractor_version: str) -> Dict[str, Any]:
    """Generate comprehensive coverage report."""
    report = {
        'city': city,
        'year': year,
        'processing_date': datetime.utcnow().isoformat() + 'Z',
        'extractor_version': extractor_version,
        'processing_time_seconds': processing_time,
        
        'topography_coverage': topography_coverage,
        'canopy_coverage': canopy_coverage,
        
        'quality_flags': {
            'dem_coverage_sufficient': topography_coverage.get('coverage_fraction', 0) >= 0.7,
            'ndvi_coverage_sufficient': canopy_coverage.get('coverage_fraction', 0) >= 0.3,
            'seasonal_processing_successful': canopy_coverage.get('seasonal_bands_available', False),
            'vector_fallback_rate_acceptable': (canopy_coverage.get('vector_fallback_used', 0) / 
                                              max(canopy_coverage.get('total_edges', 1), 1)) <= 0.5
        }
    }
    
    return report


def run_pipeline(city: str, year: int, bbox: Optional[str] = None, name: Optional[str] = None,
                force_v3: bool = False, force_v2: bool = False, config_path: str = "config/cities.yaml") -> bool:
    """Run the complete v3 environmental feature extraction pipeline."""
    start_time = time.time()
    
    # Use custom name if provided
    city_name = name if name else city
    logger.info(f"Starting v3 pipeline for {city_name} ({year})")
    
    # Load configuration
    config = load_city_config(config_path)
    params = get_city_parameters(config, city)
    
    # Determine extraction version
    use_v3 = should_use_v3(params, force_v3, force_v2)
    extractor_version = 'v3' if use_v3 else 'v2'
    logger.info(f"Using extractor version: {extractor_version}")
    
    if not use_v3:
        logger.info("v2 extraction requested - use existing v2 pipeline")
        return True
    
    # Ensure output directories
    processed_dir, reports_dir = ensure_directories(city_name)
    
    # Define file paths
    if bbox:
        # Use bbox-based processing
        network_path = f"data/processed/{city_name}/gpkg/street_network_{year}_{city_name}.gpkg"
        logger.warning(f"Bbox processing not fully implemented - expecting network at {network_path}")
    else:
        network_path = f"data/processed/{city_name}/gpkg/street_network_{year}_{city_name}.gpkg"
    
    dem_path = f"data/processed/{city_name}/raster/dem_{year}_{city_name}.tif"
    ndvi_path = f"data/processed/{city_name}/raster/green_{year}_{city_name}.tif"
    
    # Alternative multi-band NDVI path
    ndvi_multiband_path = f"data/processed/{city_name}/raster/green_{year}_{city_name}_multiband.tif"
    if os.path.exists(ndvi_multiband_path):
        ndvi_path = ndvi_multiband_path
        logger.info(f"Using multi-band NDVI: {ndvi_multiband_path}")
    
    # Output paths
    topo_output = processed_dir / f"topography_v3_{year}_{city_name}.csv"
    canopy_output = processed_dir / f"green_canopy_v3_{year}_{city_name}.csv"
    
    # Check required inputs exist
    missing_files = []
    for path, name in [(network_path, 'network'), (dem_path, 'DEM'), (ndvi_path, 'NDVI')]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        logger.error(f"Missing required input files:\n" + "\n".join(missing_files))
        return False
    
    # Run topography v3 extraction
    topo_success, topo_coverage = run_topography_v3(
        network_path=network_path,
        dem_path=dem_path,
        output_path=str(topo_output),
        params=params,
        emit_v2_names=True
    )
    
    # Run green canopy v3 extraction
    canopy_success, canopy_coverage = run_green_canopy_v3(
        network_path=network_path,
        ndvi_path=ndvi_path,
        output_path=str(canopy_output),
        params=params,
        emit_v2_names=True
    )
    
    # Generate coverage report
    processing_time = time.time() - start_time
    report = generate_coverage_report(
        city=city_name,
        year=year,
        topography_coverage=topo_coverage,
        canopy_coverage=canopy_coverage,
        processing_time=processing_time,
        extractor_version=extractor_version
    )
    
    # Save coverage report
    report_path = reports_dir / f"env_coverage_{city_name}_{year}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Coverage report saved to: {report_path}")
    
    # Log summary
    overall_success = topo_success and canopy_success
    logger.info(f"Pipeline completed in {processing_time:.1f}s - "
               f"Success: {overall_success}, "
               f"Topo: {topo_coverage.get('coverage_fraction', 0):.1%}, "
               f"Canopy: {canopy_coverage.get('coverage_fraction', 0):.1%}")
    
    return overall_success


def main():
    """Command-line interface for v3 environmental feature extraction."""
    parser = argparse.ArgumentParser(
        description='Run v3 environmental feature extraction pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/run_env_v3.py melbourne 2023
  python pipelines/run_env_v3.py --city melbourne --year 2023 --force-v3
  python pipelines/run_env_v3.py --bbox "-37.9,144.8,-37.7,145.1" --year 2023 --name melbourne_test
        """)
    
    parser.add_argument('city', nargs='?', help='City name (if not using --city or --bbox)')
    parser.add_argument('year', nargs='?', type=int, help='Year (if not using --year)')
    
    parser.add_argument('--city', help='City name')
    parser.add_argument('--year', type=int, help='Analysis year')
    parser.add_argument('--bbox', help='Bounding box as "minlat,minlon,maxlat,maxlon"')
    parser.add_argument('--name', help='Custom name for bbox-based processing')
    
    parser.add_argument('--force-v3', action='store_true', help='Force v3 extraction')
    parser.add_argument('--force-v2', action='store_true', help='Force v2 extraction (exit early)')
    parser.add_argument('--config', default='config/cities.yaml', help='Path to cities config file')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse arguments
    city = args.city or args.city
    year = args.year or args.year
    
    if not city and not args.bbox:
        parser.error("Must specify either city name or --bbox")
    if not year:
        parser.error("Must specify year")
    
    if args.force_v3 and args.force_v2:
        parser.error("Cannot specify both --force-v3 and --force-v2")
    
    # Run pipeline
    success = run_pipeline(
        city=city or "bbox_area",
        year=year,
        bbox=args.bbox,
        name=args.name,
        force_v3=args.force_v3,
        force_v2=args.force_v2,
        config_path=args.config
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()