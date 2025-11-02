#!/usr/bin/env python3
"""
STAC-based Raster Fetcher for City-Scale Environmental Data

Fetches DEM and NDVI rasters from cloud-native STAC catalogs:
- DEM: Copernicus GLO-30 from Microsoft Planetary Computer
- NDVI: Sentinel-2 L2A from AWS Earth Search (cloud-masked, median composite)

Outputs city-scale rasters optimized for pedestrian volume modeling:
- data/external/<city>/dem.tif (EPSG:3857, float32, ~30m resolution)  
- data/external/<city>/green.tif (EPSG:3857, NDVI float32 [-1,1], ~10m resolution)

Key features:
- Windowed processing for memory efficiency
- Quality-controlled cloud masking using SCL bands
- Seasonal vegetation composites (leaf-on periods)
- Built-in coverage validation and guardrails
- Comprehensive QC reporting with metrics

Usage:
    python scripts/get_rasters_stac.py --city melbourne --year 2023 --aoi-preset melbourne
    python scripts/get_rasters_stac.py --city melbourne --aoi-preset melbourne_cbd --smoke-test
"""
import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import pystac_client
    import stackstac
    import xarray as xr
    import rioxarray as rxr
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    from rasterio.enums import Resampling
    from shapely.geometry import box
    import pyproj
    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies. Install with: pip install -r scripts/requirements_fetch.txt")
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
WGS84_CRS = CRS.from_epsg(4326)
WEB_MERCATOR_CRS = CRS.from_epsg(3857)
DEFAULT_YEAR = 2023
DEFAULT_NDVI_BUFFER_DAYS = 90

# Cloud masking values for Sentinel-2 SCL band
# https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
SCL_CLEAR_VALUES = [4, 5, 6, 7]  # Vegetation, Not-vegetated, Water, Unclassified (clear)
SCL_CLOUD_VALUES = [1, 3, 8, 9, 10, 11]  # Saturated, Cloud shadows, Clouds, Cirrus, Snow, etc.


class STACFetchError(Exception):
    """Custom exception for STAC fetching errors."""
    pass


def load_aoi_presets(yaml_path: str = "scripts/aoi_presets.yaml") -> Dict[str, Any]:
    """Load AOI presets from YAML configuration."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise STACFetchError(f"AOI presets file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise STACFetchError(f"Error parsing AOI presets YAML: {e}")


def get_seasonal_date_range(aoi_config: Dict, year: int, 
                           start_offset: Optional[str] = None, 
                           end_offset: Optional[str] = None) -> Tuple[str, str]:
    """Generate seasonal date range for vegetation (leaf-on period)."""
    seasonal_months = aoi_config.get('seasonal_months', [6, 7, 8])
    
    if start_offset and end_offset:
        # Use explicit date range if provided
        return start_offset, end_offset
    
    # Generate leaf-on period based on seasonal months
    if seasonal_months[0] > 6:  # Southern Hemisphere (e.g., Melbourne Dec-Feb)
        start_date = f"{year}-{seasonal_months[0]:02d}-01"
        end_month = seasonal_months[-1]
        end_year = year + 1 if end_month < seasonal_months[0] else year
        end_date = f"{end_year}-{end_month:02d}-28"
    else:  # Northern Hemisphere (e.g., NYC Jun-Sep)
        start_date = f"{year}-{seasonal_months[0]:02d}-01"
        end_date = f"{year}-{seasonal_months[-1]:02d}-30"
    
    logger.info(f"Using seasonal date range: {start_date} to {end_date}")
    return start_date, end_date


def fetch_dem_stac(bbox_wgs84: List[float], output_path: str, 
                   target_crs: str = "EPSG:3857") -> Dict[str, Any]:
    """
    Fetch DEM data from Copernicus GLO-30 via Microsoft Planetary Computer.
    
    Args:
        bbox_wgs84: Bounding box in WGS84 [west, south, east, north]
        output_path: Output path for dem.tif
        target_crs: Target coordinate reference system
        
    Returns:
        Dictionary with fetch statistics and metadata
    """
    logger.info(f"Fetching DEM for bbox {bbox_wgs84}")
    
    try:
        # Connect to Planetary Computer STAC
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        
        # Search for Copernicus DEM items
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox_wgs84
        )
        
        items = list(search.items())
        logger.info(f"Found {len(items)} DEM tiles")
        
        if not items:
            raise STACFetchError("No DEM tiles found for specified bbox")
        
        # Load and mosaic DEM data using stackstac
        logger.info("Loading DEM tiles with stackstac...")
        da = stackstac.stack(
            items,
            assets=["data"],
            bbox=bbox_wgs84,
            resolution=30,  # 30m resolution
            dtype=np.float32,
            fill_value=np.nan,
            rescale=False,
        )
        
        # Handle multi-tile mosaicking
        if len(da.time) > 1:
            logger.info("Mosaicking multiple DEM tiles...")
            dem_mosaic = da.median(dim="time", skipna=True)
        else:
            dem_mosaic = da.isel(time=0)
        
        # Reproject to target CRS if needed
        if str(dem_mosaic.rio.crs) != target_crs:
            logger.info(f"Reprojecting DEM from {dem_mosaic.rio.crs} to {target_crs}")
            dem_mosaic = dem_mosaic.rio.reproject(target_crs, resampling=Resampling.bilinear)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Configure raster output options
        rio_kwargs = {
            'tiled': True,
            'compress': 'lzw',
            'predictor': 3,  # Floating point predictor
            'interleave': 'pixel',
        }
        
        # Write DEM to file with chunking to avoid memory issues
        logger.info(f"Writing DEM to {output_path}")
        dem_mosaic.rio.to_raster(output_path, **rio_kwargs)
        
        # Add overviews for faster display
        try:
            with rasterio.open(output_path, 'r+') as dst:
                dst.build_overviews([2, 4, 8, 16], Resampling.average)
                dst.update_tags(ns='rio_overview', resampling='average')
        except Exception as e:
            logger.warning(f"Could not build overviews: {e}")
        
        # Compute statistics
        valid_pixels = (~np.isnan(dem_mosaic)).sum().item()
        total_pixels = dem_mosaic.size
        coverage_pct = valid_pixels / total_pixels if total_pixels > 0 else 0
        
        dem_stats = {
            'min': float(np.nanmin(dem_mosaic).item()),
            'max': float(np.nanmax(dem_mosaic).item()),
            'mean': float(np.nanmean(dem_mosaic).item()),
            'std': float(np.nanstd(dem_mosaic).item())
        }
        
        result = {
            'status': 'success',
            'tiles_found': len(items),
            'output_path': output_path,
            'total_pixels': int(total_pixels),
            'valid_pixels': int(valid_pixels),
            'coverage_percent': round(coverage_pct * 100, 2),
            'statistics': dem_stats,
            'crs': str(dem_mosaic.rio.crs),
            'resolution': dem_mosaic.rio.resolution(),
            'bounds': dem_mosaic.rio.bounds()
        }
        
        logger.info(f"DEM fetch complete: {coverage_pct:.1%} coverage, {len(items)} tiles")
        return result
        
    except Exception as e:
        logger.error(f"DEM fetch failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'output_path': output_path
        }


def fetch_ndvi_stac(bbox_wgs84: List[float], output_path: str, 
                    start_date: str, end_date: str,
                    target_crs: str = "EPSG:3857") -> Dict[str, Any]:
    """
    Fetch NDVI from Sentinel-2 L2A via AWS Earth Search with cloud masking.
    
    Args:
        bbox_wgs84: Bounding box in WGS84 [west, south, east, north]
        output_path: Output path for green.tif
        start_date: Start date for image search (YYYY-MM-DD)
        end_date: End date for image search (YYYY-MM-DD)  
        target_crs: Target coordinate reference system
        
    Returns:
        Dictionary with fetch statistics and metadata
    """
    logger.info(f"Fetching NDVI for bbox {bbox_wgs84}, date range {start_date} to {end_date}")
    
    try:
        # Connect to Earth Search STAC
        catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
        
        # Search for Sentinel-2 L2A items
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox_wgs84,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": 80}}  # Pre-filter high cloud scenes
        )
        
        items = list(search.items())
        logger.info(f"Found {len(items)} Sentinel-2 scenes")
        
        if not items:
            raise STACFetchError("No Sentinel-2 scenes found for specified bbox and date range")
        
        # Load required assets (Red, NIR, SCL for cloud masking)
        logger.info("Loading Sentinel-2 assets (B04, B08, SCL)...")
        da = stackstac.stack(
            items,
            assets=["B04", "B08", "SCL"],  # Red, NIR, Scene Classification Layer
            bbox=bbox_wgs84,
            resolution=10,  # 10m resolution
            dtype=np.float32,
            fill_value=np.nan,
            rescale=False,
        )
        
        # Apply cloud masking using SCL band
        logger.info("Applying cloud masking...")
        scl = da.sel(band="SCL")
        
        # Create cloud mask (True for clear pixels)
        clear_mask = xr.zeros_like(scl, dtype=bool)
        for clear_val in SCL_CLEAR_VALUES:
            clear_mask = clear_mask | (scl == clear_val)
        
        # Extract and scale reflectance bands
        red = da.sel(band="B04").where(clear_mask) / 10000.0  # Scale to 0-1 reflectance
        nir = da.sel(band="B08").where(clear_mask) / 10000.0
        
        # Compute NDVI: (NIR - Red) / (NIR + Red)
        ndvi = (nir - red) / (nir + red)
        
        # Create median composite across time
        logger.info("Computing median NDVI composite...")
        ndvi_composite = ndvi.median(dim="time", skipna=True)
        
        # Reproject to target CRS if needed
        if str(ndvi_composite.rio.crs) != target_crs:
            logger.info(f"Reprojecting NDVI from {ndvi_composite.rio.crs} to {target_crs}")
            ndvi_composite = ndvi_composite.rio.reproject(target_crs, resampling=Resampling.bilinear)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Configure raster output options
        rio_kwargs = {
            'tiled': True,
            'compress': 'lzw',
            'predictor': 3,  # Floating point predictor
            'interleave': 'pixel',
        }
        
        # Write NDVI to file
        logger.info(f"Writing NDVI to {output_path}")
        ndvi_composite.rio.to_raster(output_path, **rio_kwargs)
        
        # Add overviews for faster display  
        try:
            with rasterio.open(output_path, 'r+') as dst:
                dst.build_overviews([2, 4, 8, 16], Resampling.average)
                dst.update_tags(ns='rio_overview', resampling='average')
        except Exception as e:
            logger.warning(f"Could not build overviews: {e}")
        
        # Compute statistics
        valid_pixels = (~np.isnan(ndvi_composite)).sum().item()
        total_pixels = ndvi_composite.size
        coverage_pct = valid_pixels / total_pixels if total_pixels > 0 else 0
        
        # NDVI-specific stats
        ndvi_data = ndvi_composite.where(~np.isnan(ndvi_composite))
        vegetation_pixels = (ndvi_data > 0).sum().item()
        green_pixels = (ndvi_data > 0.2).sum().item()
        vegetation_pct = vegetation_pixels / valid_pixels if valid_pixels > 0 else 0
        green_pct = green_pixels / valid_pixels if valid_pixels > 0 else 0
        
        ndvi_stats = {
            'min': float(np.nanmin(ndvi_composite).item()),
            'max': float(np.nanmax(ndvi_composite).item()),
            'mean': float(np.nanmean(ndvi_composite).item()),
            'std': float(np.nanstd(ndvi_composite).item()),
            'p50': float(np.nanpercentile(ndvi_composite, 50)),
            'p95': float(np.nanpercentile(ndvi_composite, 95)),
            'vegetation_percent': round(vegetation_pct * 100, 2),
            'green_percent': round(green_pct * 100, 2)
        }
        
        result = {
            'status': 'success',
            'scenes_found': len(items),
            'scenes_used': len([item for item in items]),  # All scenes contribute to median
            'date_range': f"{start_date} to {end_date}",
            'output_path': output_path,
            'total_pixels': int(total_pixels),
            'valid_pixels': int(valid_pixels),
            'coverage_percent': round(coverage_pct * 100, 2),
            'statistics': ndvi_stats,
            'crs': str(ndvi_composite.rio.crs),
            'resolution': ndvi_composite.rio.resolution(),
            'bounds': ndvi_composite.rio.bounds()
        }
        
        logger.info(f"NDVI fetch complete: {coverage_pct:.1%} coverage, {len(items)} scenes")
        return result
        
    except Exception as e:
        logger.error(f"NDVI fetch failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'output_path': output_path
        }


def validate_coverage(dem_result: Dict[str, Any], ndvi_result: Dict[str, Any], 
                     thresholds: Dict[str, float], city: str) -> Dict[str, Any]:
    """Validate raster coverage against quality thresholds."""
    validation = {
        'status': 'unknown',
        'dem_pass': False,
        'ndvi_pass': False,
        'issues': []
    }
    
    # Check DEM coverage
    if dem_result.get('status') == 'success':
        dem_coverage = dem_result.get('coverage_percent', 0) / 100.0
        dem_threshold = thresholds.get('dem_coverage_min', 0.8)
        
        validation['dem_pass'] = dem_coverage >= dem_threshold
        if not validation['dem_pass']:
            validation['issues'].append(
                f"DEM coverage {dem_coverage:.1%} below threshold {dem_threshold:.1%}"
            )
    else:
        validation['issues'].append(f"DEM fetch failed: {dem_result.get('error', 'Unknown')}")
    
    # Check NDVI coverage
    if ndvi_result.get('status') == 'success':
        ndvi_coverage = ndvi_result.get('coverage_percent', 0) / 100.0
        ndvi_threshold = thresholds.get('ndvi_coverage_min', 0.6)
        
        validation['ndvi_pass'] = ndvi_coverage >= ndvi_threshold
        if not validation['ndvi_pass']:
            validation['issues'].append(
                f"NDVI coverage {ndvi_coverage:.1%} below threshold {ndvi_threshold:.1%}"
            )
        
        # Check vegetation content
        veg_pct = ndvi_result.get('statistics', {}).get('vegetation_percent', 0) / 100.0
        veg_threshold = thresholds.get('ndvi_valid_min', 0.3)
        
        if veg_pct < veg_threshold:
            validation['issues'].append(
                f"Vegetation content {veg_pct:.1%} below threshold {veg_threshold:.1%}"
            )
    else:
        validation['issues'].append(f"NDVI fetch failed: {ndvi_result.get('error', 'Unknown')}")
    
    # Determine overall status
    if validation['dem_pass'] and validation['ndvi_pass'] and not validation['issues']:
        validation['status'] = 'pass'
    elif validation['dem_pass'] or validation['ndvi_pass']:
        validation['status'] = 'partial'
    else:
        validation['status'] = 'fail'
    
    return validation


def write_reports(city: str, year: int, dem_result: Dict[str, Any], 
                 ndvi_result: Dict[str, Any], validation: Dict[str, Any],
                 fetch_params: Dict[str, Any]) -> Tuple[str, str]:
    """Write fetch metrics and summary reports."""
    
    # Ensure reports directory exists
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/runs", exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON metrics report
    json_path = f"reports/metrics/{city}_{year}_fetch_metrics.json"
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'city': city,
        'year': year,
        'parameters': fetch_params,
        'dem': dem_result,
        'ndvi': ndvi_result,
        'validation': validation
    }
    
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Markdown summary report
    md_path = f"reports/runs/{timestamp}_{city}_{year}_fetch_summary.md"
    
    with open(md_path, 'w') as f:
        f.write(f"# STAC Raster Fetch Report - {city.title()} {year}\n\n")
        f.write(f"**Generated:** {metrics['timestamp']}  \n")
        f.write(f"**Status:** {validation['status'].upper()}  \n")
        f.write(f"**AOI Preset:** {fetch_params['aoi_preset']}  \n\n")
        
        # Parameters
        f.write("## Fetch Parameters\n\n")
        f.write(f"- **Bounding Box (WGS84):** {fetch_params['bbox_wgs84']}\n")
        f.write(f"- **NDVI Date Range:** {fetch_params.get('ndvi_date_range', 'N/A')}\n")
        f.write(f"- **Target CRS:** {fetch_params.get('target_crs', 'EPSG:3857')}\n\n")
        
        # DEM Results
        f.write("## DEM Results\n\n")
        if dem_result['status'] == 'success':
            f.write(f"- **Status:** SUCCESS\n")
            f.write(f"- **Coverage:** {dem_result['coverage_percent']:.1f}%\n")
            f.write(f"- **Tiles:** {dem_result['tiles_found']}\n")
            f.write(f"- **Resolution:** {dem_result['resolution']}\n")
            f.write(f"- **Statistics:** min={dem_result['statistics']['min']:.1f}m, ")
            f.write(f"max={dem_result['statistics']['max']:.1f}m, ")
            f.write(f"mean={dem_result['statistics']['mean']:.1f}m\n")
        else:
            f.write(f"- **Status:** FAILED\n")
            f.write(f"- **Error:** {dem_result.get('error', 'Unknown error')}\n")
        f.write("\n")
        
        # NDVI Results  
        f.write("## NDVI Results\n\n")
        if ndvi_result['status'] == 'success':
            f.write(f"- **Status:** SUCCESS\n")
            f.write(f"- **Coverage:** {ndvi_result['coverage_percent']:.1f}%\n")
            f.write(f"- **Scenes:** {ndvi_result['scenes_found']}\n")
            f.write(f"- **Resolution:** {ndvi_result['resolution']}\n")
            stats = ndvi_result['statistics']
            f.write(f"- **NDVI Stats:** min={stats['min']:.3f}, ")
            f.write(f"max={stats['max']:.3f}, mean={stats['mean']:.3f}\n")
            f.write(f"- **Vegetation:** {stats['vegetation_percent']:.1f}% (NDVI > 0), ")
            f.write(f"{stats['green_percent']:.1f}% (NDVI > 0.2)\n")
        else:
            f.write(f"- **Status:** FAILED\n")
            f.write(f"- **Error:** {ndvi_result.get('error', 'Unknown error')}\n")
        f.write("\n")
        
        # Validation
        f.write("## Quality Validation\n\n")
        f.write(f"- **Overall Status:** {validation['status'].upper()}\n")
        f.write(f"- **DEM Pass:** {validation['dem_pass']}\n")
        f.write(f"- **NDVI Pass:** {validation['ndvi_pass']}\n")
        
        if validation['issues']:
            f.write(f"\n**Issues:**\n")
            for issue in validation['issues']:
                f.write(f"- {issue}\n")
        
        # Output Files
        f.write("\n## Output Files\n\n")
        dem_exists = os.path.exists(dem_result.get('output_path', ''))
        ndvi_exists = os.path.exists(ndvi_result.get('output_path', ''))
        
        f.write(f"- **DEM:** `{dem_result.get('output_path', 'N/A')}` ")
        f.write(f"({'EXISTS' if dem_exists else 'MISSING'})\n")
        f.write(f"- **NDVI:** `{ndvi_result.get('output_path', 'N/A')}` ")
        f.write(f"({'EXISTS' if ndvi_exists else 'MISSING'})\n\n")
    
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch city-scale DEM and NDVI rasters from STAC catalogs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/get_rasters_stac.py --city melbourne --year 2023 --aoi-preset melbourne
  python scripts/get_rasters_stac.py --city melbourne --aoi-preset melbourne_cbd --dem-only
  python scripts/get_rasters_stac.py --city NewYork --ndvi-start 2023-07-01 --ndvi-end 2023-08-31
        """
    )
    
    parser.add_argument("--city", required=True, 
                       help="City name for output directory")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR,
                       help="Year for raster data (default: 2023)")
    parser.add_argument("--aoi-preset", required=True,
                       help="AOI preset name from aoi_presets.yaml")
    
    # Date range options
    parser.add_argument("--ndvi-start", 
                       help="NDVI start date (YYYY-MM-DD, overrides seasonal)")
    parser.add_argument("--ndvi-end",
                       help="NDVI end date (YYYY-MM-DD, overrides seasonal)")
    
    # Selective fetching
    parser.add_argument("--dem-only", action="store_true",
                       help="Only fetch DEM data")
    parser.add_argument("--ndvi-only", action="store_true", 
                       help="Only fetch NDVI data")
    
    # Testing and debugging
    parser.add_argument("--smoke-test", action="store_true",
                       help="Use relaxed quality thresholds for testing")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate exclusive options
    if args.dem_only and args.ndvi_only:
        parser.error("Cannot specify both --dem-only and --ndvi-only")
    
    try:
        # Load AOI presets
        logger.info("Loading AOI presets...")
        presets = load_aoi_presets()
        
        if args.aoi_preset not in presets:
            raise STACFetchError(f"AOI preset '{args.aoi_preset}' not found in aoi_presets.yaml")
        
        aoi_config = presets[args.aoi_preset]
        bbox_wgs84 = aoi_config['bbox']
        
        # Determine quality thresholds
        threshold_key = 'cbd_test' if args.smoke_test or '_cbd' in args.aoi_preset else 'full_city'
        thresholds = presets['quality_thresholds'][threshold_key]
        
        logger.info(f"Using AOI preset '{args.aoi_preset}': {aoi_config.get('description', 'N/A')}")
        logger.info(f"Bounding box: {bbox_wgs84}")
        logger.info(f"Quality thresholds: {threshold_key}")
        
        # Generate seasonal date range for NDVI
        ndvi_start, ndvi_end = get_seasonal_date_range(
            aoi_config, args.year, args.ndvi_start, args.ndvi_end
        )
        
        # Prepare output paths
        output_dir = f"data/external/{args.city}"
        dem_path = os.path.join(output_dir, "dem.tif")
        ndvi_path = os.path.join(output_dir, "green.tif")
        
        # Store fetch parameters for reporting
        fetch_params = {
            'city': args.city,
            'year': args.year,
            'aoi_preset': args.aoi_preset,
            'bbox_wgs84': bbox_wgs84,
            'ndvi_date_range': f"{ndvi_start} to {ndvi_end}",
            'target_crs': "EPSG:3857",
            'threshold_mode': threshold_key,
            'dem_only': args.dem_only,
            'ndvi_only': args.ndvi_only
        }
        
        # Initialize results
        dem_result = {'status': 'skipped', 'output_path': dem_path}
        ndvi_result = {'status': 'skipped', 'output_path': ndvi_path}
        
        # Fetch DEM if requested
        if not args.ndvi_only:
            logger.info("Starting DEM fetch...")
            dem_result = fetch_dem_stac(bbox_wgs84, dem_path)
        
        # Fetch NDVI if requested  
        if not args.dem_only:
            logger.info("Starting NDVI fetch...")
            ndvi_result = fetch_ndvi_stac(bbox_wgs84, ndvi_path, ndvi_start, ndvi_end)
        
        # Validate results
        validation = validate_coverage(dem_result, ndvi_result, thresholds, args.city)
        
        # Write reports
        json_report, md_report = write_reports(
            args.city, args.year, dem_result, ndvi_result, validation, fetch_params
        )
        
        # Print summary
        print(f"\n=== STAC FETCH SUMMARY ===")
        print(f"City: {args.city}")
        print(f"Year: {args.year}")
        print(f"AOI: {args.aoi_preset}")
        print(f"Status: {validation['status'].upper()}")
        
        if dem_result['status'] == 'success':
            print(f"DEM: {dem_result['coverage_percent']:.1f}% coverage ({dem_result['tiles_found']} tiles)")
        elif dem_result['status'] != 'skipped':
            print(f"DEM: FAILED ({dem_result.get('error', 'Unknown error')})")
        
        if ndvi_result['status'] == 'success':
            print(f"NDVI: {ndvi_result['coverage_percent']:.1f}% coverage ({ndvi_result['scenes_found']} scenes)")
            veg_pct = ndvi_result['statistics']['vegetation_percent']
            print(f"Vegetation: {veg_pct:.1f}% of valid pixels")
        elif ndvi_result['status'] != 'skipped':
            print(f"NDVI: FAILED ({ndvi_result.get('error', 'Unknown error')})")
        
        print(f"\nReports:")
        print(f"- JSON: {json_report}")
        print(f"- Markdown: {md_report}")
        
        if validation['issues']:
            print(f"\nIssues:")
            for issue in validation['issues']:
                print(f"- {issue}")
        
        # Exit with appropriate code
        if validation['status'] == 'fail':
            print(f"\nFetch failed quality validation - exiting with error")
            sys.exit(1)
        elif validation['status'] == 'partial':
            print(f"\nPartial success - some quality issues detected")
        else:
            print(f"\nFetch completed successfully!")
        
    except STACFetchError as e:
        logger.error(f"STAC fetch error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Fetch interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()