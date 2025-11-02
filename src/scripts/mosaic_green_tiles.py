#!/usr/bin/env python3
"""
NDVI Tile Mosaicking Script

Mosaics multiple NDVI tile rasters into a single seamless raster.
All input tiles must share the same CRS.

Usage:
  python scripts/mosaic_green_tiles.py \
    --inputs data/external/melbourne/tiles/*.tif \
    --out data/external/melbourne/green.tif \
    --nodata -9999 \
    --method max

Requirements: pip install rasterio numpy
"""

import argparse
import glob
import os
import sys
from pathlib import Path

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install rasterio numpy")
    sys.exit(1)


def validate_input_crs(input_paths: list) -> str:
    """
    Validate that all input rasters have the same CRS.
    
    Args:
        input_paths: List of raster file paths
        
    Returns:
        Common CRS string
        
    Raises:
        ValueError: If CRS don't match or files can't be read
    """
    if not input_paths:
        raise ValueError("No input files provided")
    
    common_crs = None
    crs_info = []
    
    for path in input_paths:
        try:
            with rasterio.open(path) as src:
                crs = str(src.crs)
                crs_info.append(f"{Path(path).name}: {crs}")
                
                if common_crs is None:
                    common_crs = crs
                elif crs != common_crs:
                    error_msg = "ERROR: Input rasters have mismatched CRS:\n"
                    error_msg += "\n".join(crs_info)
                    error_msg += "\n\nAll input tiles must share the same coordinate system."
                    error_msg += "\nReproject tiles to a common CRS before mosaicking."
                    raise ValueError(error_msg)
                    
        except rasterio.RasterioIOError as e:
            raise ValueError(f"Cannot read raster file {path}: {e}")
    
    print(f"All {len(input_paths)} input tiles share CRS: {common_crs}")
    return common_crs


def mosaic_tiles(input_paths: list, output_path: str, nodata_value: float, method: str):
    """
    Mosaic multiple raster tiles into a single output.
    
    Args:
        input_paths: List of input raster paths
        output_path: Output raster path
        nodata_value: NoData value to use
        method: Mosaic method ('first', 'last', 'min', 'max', 'mean')
    """
    print(f"Mosaicking {len(input_paths)} tiles using method '{method}'")
    
    # Open all input rasters
    src_files_to_mosaic = []
    try:
        for path in input_paths:
            src = rasterio.open(path)
            src_files_to_mosaic.append(src)
            print(f"  - {Path(path).name}")
        
        # Perform mosaic
        print("Computing mosaic...")
        mosaic_array, mosaic_transform = merge(
            src_files_to_mosaic,
            method=method,
            nodata=nodata_value
        )
        
        # Get metadata from first source
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        # Update metadata for mosaic
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic_array.shape[1],
            "width": mosaic_array.shape[2],
            "transform": mosaic_transform,
            "nodata": nodata_value,
            "dtype": "float32",
            "compress": "lzw",
            "tiled": True
        })
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write mosaic
        print(f"Writing mosaic to {output_path}")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic_array)
        
        # Print summary statistics
        print_mosaic_summary(mosaic_array, nodata_value, out_meta, output_path)
        
    finally:
        # Close all source files
        for src in src_files_to_mosaic:
            src.close()


def print_mosaic_summary(mosaic_array: np.ndarray, nodata_value: float, 
                        metadata: dict, output_path: Path):
    """Print mosaic processing summary."""
    # Calculate statistics
    valid_mask = mosaic_array[0] != nodata_value
    total_pixels = mosaic_array[0].size
    valid_pixels = np.sum(valid_mask)
    valid_percentage = (valid_pixels / total_pixels) * 100
    
    if valid_pixels > 0:
        data_min = np.min(mosaic_array[0][valid_mask])
        data_max = np.max(mosaic_array[0][valid_mask])
    else:
        data_min = data_max = np.nan
    
    print(f"\n=== Mosaic Summary ===")
    print(f"Output: {output_path}")
    print(f"CRS: {metadata['crs']}")
    print(f"Size: {metadata['width']} x {metadata['height']} pixels")
    print(f"Valid pixels: {valid_percentage:.1f}% ({valid_pixels:,} / {total_pixels:,})")
    print(f"Value range: {data_min:.3f} to {data_max:.3f}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Mosaic multiple NDVI tile rasters into one seamless raster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/mosaic_green_tiles.py ^
    --inputs data/external/melbourne/tiles/*.tif ^
    --out data/external/melbourne/green.tif ^
    --nodata -9999 ^
    --method max

Mosaic methods:
  - first: Use value from first overlapping raster
  - last: Use value from last overlapping raster  
  - min: Use minimum value in overlapping areas
  - max: Use maximum value in overlapping areas
  - mean: Use mean value in overlapping areas
        """
    )
    
    parser.add_argument('--inputs', required=True,
                       help='Input raster pattern (e.g., "tiles/*.tif")')
    parser.add_argument('--out', required=True,
                       help='Output mosaic raster path')
    parser.add_argument('--nodata', type=float, default=-9999,
                       help='NoData value (default: -9999)')
    parser.add_argument('--method', choices=['first', 'last', 'min', 'max', 'mean'],
                       default='max', help='Mosaic method (default: max)')
    
    args = parser.parse_args()
    
    try:
        # Expand glob pattern
        input_files = glob.glob(args.inputs)
        
        if not input_files:
            print(f"ERROR: No files found matching pattern: {args.inputs}")
            return 1
        
        if len(input_files) == 1:
            print(f"WARNING: Only one input file found. Consider copying instead of mosaicking.")
        
        print(f"Found {len(input_files)} input files")
        
        # Validate CRS consistency
        common_crs = validate_input_crs(input_files)
        
        # Perform mosaic
        mosaic_tiles(input_files, args.out, args.nodata, args.method)
        
        print("\nMosaicking completed successfully!")
        return 0
        
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())