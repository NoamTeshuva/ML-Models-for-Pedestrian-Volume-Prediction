#!/usr/bin/env python3
"""
Sentinel-2 NDVI Processing Script

Converts Sentinel-2 L2A data into cloud-masked NDVI GeoTIFF using pure Python.
Reads B04 (Red), B08 (NIR), and SCL (Scene Classification) bands, computes NDVI,
masks clouds/shadows, and outputs to GeoTIFF format.

Requirements: pip install rasterio numpy
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install rasterio numpy")
    sys.exit(1)


def validate_file_exists(filepath: str, band_name: str) -> Path:
    """Validate that input file exists and is readable."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"{band_name} file not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"{band_name} path is not a file: {filepath}")
    return path


def read_band_as_float32(filepath: Path, band_name: str):
    """Read a rasterio band as float32 with metadata."""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            return data, profile
    except Exception as e:
        raise RuntimeError(f"Failed to read {band_name} from {filepath}: {e}")


def upsample_scl_to_10m(scl_path: Path, target_profile: dict):
    """Upsample 20m SCL to 10m grid using nearest neighbor resampling."""
    with rasterio.open(scl_path) as scl_src:
        scl_data = scl_src.read(1)
        scl_profile = scl_src.profile
        
        # Create destination array with target dimensions
        dst_data = np.empty((target_profile['height'], target_profile['width']), dtype=np.uint8)
        
        # Reproject SCL to match B04/B08 grid
        reproject(
            source=scl_data,
            destination=dst_data,
            src_transform=scl_profile['transform'],
            src_crs=scl_profile['crs'],
            dst_transform=target_profile['transform'],
            dst_crs=target_profile['crs'],
            resampling=Resampling.nearest
        )
        
        return dst_data


def compute_ndvi_with_cloud_mask(b04_data: np.ndarray, b08_data: np.ndarray, scl_data: np.ndarray) -> np.ndarray:
    """
    Compute NDVI and apply cloud/shadow masking.
    
    Cloud/shadow SCL classes to mask: {3, 8, 9, 10, 11}
    - 3: Cloud shadows
    - 8: Cloud medium probability
    - 9: Cloud high probability
    - 10: Thin cirrus
    - 11: Snow/ice
    """
    # Convert to float32 for computation
    b04 = b04_data.astype(np.float32)
    b08 = b08_data.astype(np.float32)
    
    # Compute NDVI with small epsilon to avoid division by zero
    epsilon = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + epsilon)
    
    # Create cloud/shadow mask
    cloud_shadow_classes = {3, 8, 9, 10, 11}
    cloud_mask = np.isin(scl_data, list(cloud_shadow_classes))
    
    # Apply mask - set masked areas to NoData
    ndvi_masked = ndvi.copy()
    ndvi_masked[cloud_mask] = -9999.0
    
    return ndvi_masked


def write_ndvi_geotiff(ndvi_data: np.ndarray, profile: dict, output_path: Path):
    """Write NDVI data to GeoTIFF with proper settings."""
    # Update profile for output - force GeoTIFF driver
    output_profile = profile.copy()
    output_profile.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': -9999.0,
        'compress': 'lzw',
        'tiled': True,
        'count': 1
    })
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(ndvi_data, 1)


def print_summary(ndvi_data: np.ndarray, profile: dict, output_path: Path):
    """Print processing summary."""
    # Calculate statistics for valid pixels only
    valid_mask = ndvi_data != -9999.0
    valid_pixels = np.sum(valid_mask)
    total_pixels = ndvi_data.size
    valid_percentage = (valid_pixels / total_pixels) * 100
    
    if valid_pixels > 0:
        ndvi_min = np.min(ndvi_data[valid_mask])
        ndvi_max = np.max(ndvi_data[valid_mask])
    else:
        ndvi_min = ndvi_max = np.nan
    
    print(f"=== NDVI Processing Summary ===")
    print(f"Output: {output_path}")
    print(f"CRS: {profile['crs']}")
    print(f"Raster size: {profile['width']} x {profile['height']} pixels")
    print(f"Valid pixels: {valid_percentage:.1f}% ({valid_pixels:,} / {total_pixels:,})")
    print(f"NDVI range: {ndvi_min:.3f} to {ndvi_max:.3f}")
    print(f"Pixel size: {abs(profile['transform'][0]):.1f}m")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Sentinel-2 L2A bands to cloud-masked NDVI GeoTIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/make_green_ndvi.py \\
    --b04 "C:/Data/S2/.../IMG_DATA/R10m/*B04_10m.jp2" \\
    --b08 "C:/Data/S2/.../IMG_DATA/R10m/*B08_10m.jp2" \\
    --scl "C:/Data/S2/.../IMG_DATA/R20m/*SCL_20m.jp2" \\
    --out "data/external/melbourne/green.tif"
        """
    )
    
    parser.add_argument('--b04', required=True, help='Path to B04 (Red) band JP2 file')
    parser.add_argument('--b08', required=True, help='Path to B08 (NIR) band JP2 file')
    parser.add_argument('--scl', required=True, help='Path to SCL (Scene Classification) band JP2 file')
    parser.add_argument('--out', required=True, help='Output path for NDVI GeoTIFF')
    parser.add_argument('--out-tile-name', default='', help='Optional tile name for organized output (e.g., T55HCU)')
    
    args = parser.parse_args()
    
    try:
        # Validate input files
        b04_path = validate_file_exists(args.b04, "B04 (Red)")
        b08_path = validate_file_exists(args.b08, "B08 (NIR)")
        scl_path = validate_file_exists(args.scl, "SCL")
        
        # Handle tile-specific output path if requested
        if args.out_tile_name:
            output_dir = Path(args.out).parent / "tiles"
            output_path = output_dir / f"green_{args.out_tile_name}.tif"
        else:
            output_path = Path(args.out)
        
        print("Reading B04 (Red) band...")
        b04_data, b04_profile = read_band_as_float32(b04_path, "B04")
        
        print("Reading B08 (NIR) band...")
        b08_data, b08_profile = read_band_as_float32(b08_path, "B08")
        
        # Validate that B04 and B08 have matching grids
        if (b04_profile['crs'] != b08_profile['crs'] or 
            b04_profile['transform'] != b08_profile['transform'] or
            b04_profile['width'] != b08_profile['width'] or
            b04_profile['height'] != b08_profile['height']):
            raise ValueError(
                "B04 and B08 bands have mismatched grids. "
                "Ensure both are 10m L2A bands from the same granule."
            )
        
        print("Upsampling SCL (Scene Classification) from 20m to 10m...")
        scl_data = upsample_scl_to_10m(scl_path, b08_profile)
        
        print("Computing NDVI with cloud/shadow masking...")
        ndvi_masked = compute_ndvi_with_cloud_mask(b04_data, b08_data, scl_data)
        
        print(f"Writing NDVI GeoTIFF to {output_path}...")
        write_ndvi_geotiff(ndvi_masked, b08_profile, output_path)
        
        print_summary(ndvi_masked, b08_profile, output_path)
        print("\nProcessing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        if "mismatched grids" in str(e):
            print("HINT: Ensure both B04 and B08 are 10m L2A bands from the same granule.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()