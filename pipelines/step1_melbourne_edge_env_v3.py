#!/usr/bin/env python3
"""
Melbourne Edge-Level Environmental Features Pipeline (v3)

One-shot script to:
1. Validate inputs (network, DEM, NDVI)
2. Run diagnostics on raster-network alignment
3. Extract topography and canopy features with v3 extractors
4. Compute coverage statistics and validation
5. Generate JSON + Markdown reports
6. Exit non-zero if coverage is insufficient

Additive-only: creates new files without modifying existing ones.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines._util_yaml import load_v3_config, get_topo_params, get_canopy_params

# Color output for better UX
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def colored(text: str, color: str) -> str:
        return f"{color}{text}{Style.RESET_ALL}"
except ImportError:
    def colored(text: str, color: str) -> str:
        return text


def validate_inputs(network_path: str, dem_path: str, ndvi_path: str) -> None:
    """Validate that all required input files exist."""
    missing = []
    for name, path in [("Network", network_path), ("DEM", dem_path), ("NDVI", ndvi_path)]:
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    
    if missing:
        print(colored("FATAL: Missing required input files:", Fore.RED))
        for item in missing:
            print(colored(f"  - {item}", Fore.RED))
        sys.exit(1)
    
    print(colored("Input validation passed", Fore.GREEN))


def run_diagnostics(network_path: str, dem_path: str, ndvi_path: str, 
                   city: str, year: int) -> Dict[str, Any]:
    """Run raster-network diagnostics and parse results."""
    print(colored("Running raster-network diagnostics...", Fore.CYAN))
    
    # Ensure output directories exist
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/runs", exist_ok=True)
    
    diag_json = f"reports/metrics/raster_diag_{city}_{year}.json"
    diag_md = f"reports/runs/raster_diag_{city}_{year}.md"
    
    try:
        cmd = [
            sys.executable, "scripts/raster_network_diagnostics.py",
            "--city", city,
            "--year", str(year),
            "--network-gpkg", network_path,
            "--dem", dem_path, 
            "--ndvi", ndvi_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(colored("Diagnostics completed", Fore.GREEN))
        
        # Parse output and create basic JSON report
        diag_output = result.stdout
        basic_result = {
            "status": "completed", 
            "output": diag_output[:1000],  # First 1000 chars
            "city": city,
            "year": year
        }
        
        # Write simplified reports
        with open(diag_json, 'w') as f:
            json.dump(basic_result, f, indent=2)
        with open(diag_md, 'w') as f:
            f.write(f"# Raster Network Diagnostics - {city} {year}\n\n")
            f.write(f"```\n{diag_output}\n```\n")
        
        return basic_result
            
    except subprocess.CalledProcessError as e:
        print(colored(f"Diagnostics failed: {e}", Fore.YELLOW))
        print(f"stderr: {e.stderr}")
        return {"status": "failed", "error": str(e)}
    except FileNotFoundError:
        print(colored("Diagnostics script not found, continuing without diagnostics", Fore.YELLOW))
        return {"status": "skipped", "reason": "script not found"}


def run_topography_v3(network_path: str, dem_path: str, config: Dict[str, Any], 
                     out_csv: str, city: str) -> bool:
    """Run topography v3 feature extraction."""
    print(colored("Extracting topography features (v3)...", Fore.CYAN))
    
    topo_params = get_topo_params(config)
    
    try:
        cmd = [
            sys.executable, "src/feature_engineering/topography_features_v3.py",
            "--city", city,
            "--network-gpkg", network_path,
            "--dem", dem_path,
            "--out-csv", out_csv,
            "--emit-v2-names",
            "--max-seg-m", str(topo_params["max_seg_m"]),
            "--smooth-window", str(topo_params["smooth_window"]),
            "--dem-nodata", str(topo_params["dem_nodata"]),
            "--min-valid-fraction", str(topo_params["min_valid_fraction"]),
            "--preflight-min-coverage", "0.30",  # New: preflight coverage check
            "--auto-adjust-seg"  # New: auto segment length adjustment
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(colored(f"Topography CSV written: {out_csv}", Fore.GREEN))
            return True
        else:
            print(colored(f"Topography CSV missing or empty: {out_csv}", Fore.RED))
            return False
            
    except subprocess.CalledProcessError as e:
        print(colored(f"Topography extraction failed: {e}", Fore.RED))
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(colored("Topography v3 script not found", Fore.RED))
        return False


def run_canopy_v3(network_path: str, ndvi_path: str, config: Dict[str, Any], 
                 out_csv: str, city: str) -> bool:
    """Run canopy v3 feature extraction."""
    print(colored("Extracting canopy features (v3)...", Fore.CYAN))
    
    canopy_params = get_canopy_params(config)
    
    try:
        cmd = [
            sys.executable, "src/feature_engineering/green_canopy_features_v3.py",
            "--city", city,
            "--network-gpkg", network_path,
            "--green-raster", ndvi_path,  # Updated parameter name
            "--out-csv", out_csv,
            "--emit-v2-names",
            "--buffer-m", str(canopy_params["buffer_m"]),
            "--ndvi-threshold", str(canopy_params["ndvi_threshold"]),
            "--ndvi-rescale", str(canopy_params["ndvi_rescale"]),
            "--min-coverage", str(canopy_params["min_valid_fraction"]),  # Updated parameter name
            "--preflight-min-coverage", "0.30",  # New: preflight coverage check
            "--auto-ndvi-threshold"  # New: auto threshold detection
        ]
        
        if canopy_params["treat_zero_as_nodata"]:
            cmd.append("--treat-zero-as-nodata")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(colored(f"Canopy CSV written: {out_csv}", Fore.GREEN))
            return True
        else:
            print(colored(f"Canopy CSV missing or empty: {out_csv}", Fore.RED))
            return False
            
    except subprocess.CalledProcessError as e:
        print(colored(f"Canopy extraction failed: {e}", Fore.RED))
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(colored("Canopy v3 script not found", Fore.RED))
        return False


def compute_coverage_stats(topo_csv: str, canopy_csv: str) -> Dict[str, Any]:
    """Compute coverage statistics for extracted features."""
    print(colored("Computing coverage statistics...", Fore.CYAN))
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "files": {
            "topography": topo_csv,
            "canopy": canopy_csv
        },
        "coverage": {},
        "row_counts": {},
        "status": "unknown"
    }
    
    # Define expected columns
    topo_cols = ["mean_grade_pct", "max_grade_pct", "elev_range_m"]
    canopy_cols = ["canopy_pct_buffer"]
    
    try:
        # Analyze topography file
        if os.path.exists(topo_csv):
            df_topo = pd.read_csv(topo_csv)
            stats["row_counts"]["topography"] = len(df_topo)
            
            for col in topo_cols:
                if col in df_topo.columns:
                    non_null_rate = df_topo[col].notna().mean()
                    stats["coverage"][col] = round(non_null_rate, 4)
                else:
                    stats["coverage"][col] = 0.0
        else:
            stats["row_counts"]["topography"] = 0
            for col in topo_cols:
                stats["coverage"][col] = 0.0
        
        # Analyze canopy file
        if os.path.exists(canopy_csv):
            df_canopy = pd.read_csv(canopy_csv)
            stats["row_counts"]["canopy"] = len(df_canopy)
            
            for col in canopy_cols:
                if col in df_canopy.columns:
                    non_null_rate = df_canopy[col].notna().mean()
                    stats["coverage"][col] = round(non_null_rate, 4)
                else:
                    stats["coverage"][col] = 0.0
        else:
            stats["row_counts"]["canopy"] = 0
            for col in canopy_cols:
                stats["coverage"][col] = 0.0
    
    except Exception as e:
        print(colored(f"Coverage analysis failed: {e}", Fore.YELLOW))
        stats["error"] = str(e)
    
    return stats


def write_reports(stats: Dict[str, Any], city: str, year: int) -> Tuple[str, str]:
    """Write JSON and Markdown coverage reports."""
    json_path = f"reports/metrics/env_edge_coverage_{city}_{year}.json"
    md_path = f"reports/runs/env_edge_coverage_{city}_{year}.md"
    
    # Write JSON report
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Write Markdown report
    with open(md_path, 'w') as f:
        f.write(f"# Environmental Edge Features Coverage Report\n\n")
        f.write(f"**City:** {city}  \n")
        f.write(f"**Year:** {year}  \n")
        f.write(f"**Generated:** {stats['timestamp']}  \n\n")
        
        f.write("## Row Counts\n\n")
        for file_type, count in stats["row_counts"].items():
            f.write(f"- **{file_type.title()}**: {count:,} rows\n")
        
        f.write("\n## Coverage Statistics\n\n")
        f.write("| Feature | Non-Null Rate | Status |\n")
        f.write("|---------|---------------|--------|\n")
        
        for col, coverage in stats["coverage"].items():
            status = "Good" if coverage >= 0.3 else ("Low" if coverage >= 0.15 else "Poor")
            f.write(f"| {col} | {coverage:.1%} | {status} |\n")
        
        f.write("\n## Files Generated\n\n")
        for file_type, path in stats["files"].items():
            exists = "YES" if os.path.exists(path) else "NO"
            f.write(f"- {exists} **{file_type.title()}**: `{path}`\n")
    
    return json_path, md_path


def validate_coverage(stats: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Validate that coverage meets minimum requirements."""
    print(colored("Validating coverage requirements...", Fore.CYAN))
    
    min_topo_coverage = 0.3
    min_canopy_coverage = 0.15
    
    coverage = stats["coverage"]
    issues = []
    
    # Check topography coverage
    topo_coverage = coverage.get("mean_grade_pct", 0.0)
    if topo_coverage < min_topo_coverage:
        issues.append(f"Topography coverage too low: {topo_coverage:.1%} < {min_topo_coverage:.1%}")
    
    # Check canopy coverage
    canopy_coverage = coverage.get("canopy_pct_buffer", 0.0)
    if canopy_coverage < min_canopy_coverage:
        issues.append(f"Canopy coverage too low: {canopy_coverage:.1%} < {min_canopy_coverage:.1%}")
    
    if issues:
        print(colored("COVERAGE VALIDATION FAILED:", Fore.RED))
        for issue in issues:
            print(colored(f"  - {issue}", Fore.RED))
        
        print(colored("\nSuggested fixes:", Fore.YELLOW))
        print("  - Check NDVI scaling and thresholds in config")
        print("  - Verify raster-network spatial alignment")
        print("  - Review DEM nodata values and projection")
        print("  - Consider adjusting buffer sizes or smoothing parameters")
        
        return False
    
    print(colored("Coverage validation passed", Fore.GREEN))
    return True


def main():
    parser = argparse.ArgumentParser(description="Melbourne Edge Environmental Features Pipeline (v3)")
    parser.add_argument("--city", default="melbourne", help="City name (default: melbourne)")
    parser.add_argument("--year", type=int, default=2023, help="Year (default: 2023)")
    parser.add_argument("--network-gpkg", default="data/osm/melbourne_street_network/melbourne_network.gpkg", 
                       help="Network GeoPackage path")
    parser.add_argument("--dem", default="data/external/melbourne/dem.tif", help="DEM raster path")
    parser.add_argument("--ndvi", default="data/external/melbourne/green.tif", help="NDVI raster path")
    
    args = parser.parse_args()
    
    print(colored("Melbourne Edge Environmental Features Pipeline (v3)", Fore.MAGENTA))
    print(f"City: {args.city} | Year: {args.year}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load configuration
    config = load_v3_config(args.city)
    
    # Validate inputs
    validate_inputs(args.network_gpkg, args.dem, args.ndvi)
    
    # Prepare output paths
    os.makedirs(f"data/processed/{args.city}/csv", exist_ok=True)
    topo_csv = f"data/processed/{args.city}/csv/topography_{args.year}_{args.city}.csv"
    canopy_csv = f"data/processed/{args.city}/csv/green_canopy_{args.year}_{args.city}.csv"
    
    # Step 1: Run diagnostics (optional)
    diag_results = run_diagnostics(args.network_gpkg, args.dem, args.ndvi, args.city, args.year)
    
    # Step 2: Extract topography features
    topo_success = run_topography_v3(args.network_gpkg, args.dem, config, topo_csv, args.city)
    
    # Step 3: Extract canopy features
    canopy_success = run_canopy_v3(args.network_gpkg, args.ndvi, config, canopy_csv, args.city)
    
    # Check if both extractions succeeded
    if not (topo_success and canopy_success):
        print(colored("PIPELINE FAILED: Feature extraction incomplete", Fore.RED))
        sys.exit(1)
    
    # Step 4: Compute coverage statistics
    stats = compute_coverage_stats(topo_csv, canopy_csv)
    
    # Step 5: Write reports
    json_report, md_report = write_reports(stats, args.city, args.year)
    print(colored("Reports written:", Fore.GREEN))
    print(f"  - JSON: {json_report}")
    print(f"  - Markdown: {md_report}")
    
    # Step 6: Validate coverage and fail if insufficient
    coverage_ok = validate_coverage(stats, config)
    
    # Final summary
    print()
    print(colored("Output files created:", Fore.GREEN))
    print(f"  - {topo_csv}")
    print(f"  - {canopy_csv}")
    
    if not coverage_ok:
        print(colored("PIPELINE FAILED: Insufficient coverage", Fore.RED))
        sys.exit(1)
    
    print(colored("PIPELINE COMPLETED SUCCESSFULLY", Fore.GREEN))
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()