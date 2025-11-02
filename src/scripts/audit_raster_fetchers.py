#!/usr/bin/env python3
"""
Raster Fetcher Audit Script

Audits the repository for existing raster fetch/build capabilities:
- Code presence (scripts, imports, dependencies)
- CLI dry-runs (--help only, no network calls)
- Data presence and metadata validation
- Configuration files
- Generates JSON and Markdown reports

No network calls or file modifications - audit only.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Standard library only for core functionality
import importlib.util


def safe_import_check(module_name: str) -> bool:
    """Check if a module can be imported without actually importing it."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def check_file_presence() -> Dict[str, Any]:
    """Check for presence of raster fetcher files."""
    files_to_check = [
        "scripts/get_rasters_stac.py",
        "scripts/build_melbourne_rasters.py", 
        "scripts/get_ndvi_*_stac.py",
        "scripts/get_dem_*_pc.py",
        "pipelines/fetch_rasters.ps1",
        "scripts/aoi_presets.yaml"
    ]
    
    result = {
        "files_found": [],
        "files_missing": [],
        "glob_matches": {}
    }
    
    for file_pattern in files_to_check:
        if '*' in file_pattern:
            # Handle glob patterns
            matches = glob.glob(file_pattern)
            result["glob_matches"][file_pattern] = matches
            if matches:
                result["files_found"].extend(matches)
            else:
                result["files_missing"].append(file_pattern)
        else:
            # Handle exact file paths
            if os.path.exists(file_pattern):
                result["files_found"].append(file_pattern)
            else:
                result["files_missing"].append(file_pattern)
    
    return result


def check_imports_in_codebase() -> Dict[str, Any]:
    """Grep codebase for raster-related imports."""
    target_imports = [
        "pystac_client", "stackstac", "rioxarray", "rasterio", 
        "xarray", "WarpedVRT", "COG", "SCL"
    ]
    
    result = {
        "imports_found": {},
        "files_with_imports": [],
        "total_matches": 0
    }
    
    # Search Python files for imports
    python_files = glob.glob("**/*.py", recursive=True)
    
    for import_name in target_imports:
        matches = []
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Look for import statements
                    import_patterns = [
                        f"import {import_name}",
                        f"from {import_name}",
                        f"import.*{import_name}",
                        f"'{import_name}'",
                        f'"{import_name}"'
                    ]
                    
                    for pattern in import_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            matches.append(py_file)
                            if py_file not in result["files_with_imports"]:
                                result["files_with_imports"].append(py_file)
                            break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if matches:
            result["imports_found"][import_name] = matches
            result["total_matches"] += len(matches)
    
    return result


def check_dependencies() -> Dict[str, Any]:
    """Check requirements files for raster dependencies."""
    req_files = ["requirements.txt", "pedestrian-api/requirements.txt"]
    target_deps = [
        "pystac-client", "stackstac", "rioxarray", "rasterio", 
        "xarray", "gdal", "pyproj"
    ]
    
    result = {
        "requirements_files": {},
        "dependencies_found": [],
        "dependencies_missing": [],
        "import_tests": {}
    }
    
    # Check requirements files
    for req_file in req_files:
        if os.path.exists(req_file):
            try:
                with open(req_file, 'r') as f:
                    content = f.read()
                    found_deps = []
                    for dep in target_deps:
                        if dep in content.lower():
                            found_deps.append(dep)
                    result["requirements_files"][req_file] = found_deps
            except Exception as e:
                result["requirements_files"][req_file] = f"Error reading: {e}"
    
    # Test imports (lightweight, no network calls)
    for dep in target_deps:
        # Convert requirement name to module name
        module_name = dep.replace("-", "_")
        if module_name == "gdal":
            module_name = "osgeo.gdal"
        
        can_import = safe_import_check(module_name)
        result["import_tests"][dep] = can_import
        
        if can_import:
            result["dependencies_found"].append(dep)
        else:
            result["dependencies_missing"].append(dep)
    
    return result


def run_cli_dry_runs() -> Dict[str, Any]:
    """Run --help on detected fetch scripts (no network calls)."""
    result = {
        "cli_tests": {},
        "successful_helps": [],
        "failed_helps": []
    }
    
    scripts_to_test = [
        "scripts/get_rasters_stac.py",
        "scripts/build_melbourne_rasters.py"
    ]
    
    for script_path in scripts_to_test:
        if os.path.exists(script_path):
            try:
                cmd_result = subprocess.run(
                    [sys.executable, script_path, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                help_text = cmd_result.stdout[:500]  # First 500 chars
                result["cli_tests"][script_path] = {
                    "exit_code": cmd_result.returncode,
                    "help_text": help_text,
                    "error": cmd_result.stderr[:200] if cmd_result.stderr else None
                }
                
                if cmd_result.returncode == 0:
                    result["successful_helps"].append(script_path)
                else:
                    result["failed_helps"].append(script_path)
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                result["cli_tests"][script_path] = {
                    "exit_code": -1,
                    "help_text": None,
                    "error": str(e)
                }
                result["failed_helps"].append(script_path)
        else:
            result["cli_tests"][script_path] = {
                "exit_code": -404,
                "help_text": None,
                "error": "File not found"
            }
    
    # Check PowerShell script
    ps_script = "pipelines/fetch_rasters.ps1"
    if os.path.exists(ps_script):
        try:
            with open(ps_script, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:40]  # First 40 lines
                non_empty = [line.strip() for line in lines if line.strip()]
                result["cli_tests"][ps_script] = {
                    "exit_code": 0,
                    "help_text": "\n".join(non_empty),
                    "error": None
                }
                result["successful_helps"].append(ps_script)
        except Exception as e:
            result["cli_tests"][ps_script] = {
                "exit_code": -1,
                "help_text": None,
                "error": str(e)
            }
    
    return result


def check_raster_metadata(raster_path: str) -> Dict[str, Any]:
    """Check raster metadata using rasterio (metadata only, no full read)."""
    try:
        # Try to import rasterio
        import rasterio
        from rasterio import crs
        
        with rasterio.open(raster_path) as src:
            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]) if src.dtypes else None,
                "crs": str(src.crs) if src.crs else None,
                "nodata": src.nodata,
                "transform": str(src.transform),
                "overviews": len(src.overviews(1)) if src.count > 0 else 0,
                "pixel_count": src.width * src.height,
                "file_size": os.path.getsize(raster_path)
            }
            
            # Quality checks
            is_large_enough = metadata["pixel_count"] > 1_000_000
            is_web_mercator = metadata["crs"] and "3857" in metadata["crs"]
            is_float_dtype = metadata["dtype"] and "float" in metadata["dtype"]
            
            metadata["quality_flags"] = {
                "sufficient_size": is_large_enough,
                "web_mercator": is_web_mercator,
                "float_dtype": is_float_dtype,
                "overall_ok": is_large_enough and is_web_mercator
            }
            
            return metadata
            
    except ImportError:
        return {"error": "rasterio not available", "import_error": True}
    except Exception as e:
        return {"error": str(e), "file_error": True}


def check_data_presence(cities: List[str]) -> Dict[str, Any]:
    """Check raster data presence and metadata for each city."""
    result = {
        "cities": {},
        "summary": {
            "cities_with_both_rasters": [],
            "cities_with_partial_rasters": [],
            "cities_missing_rasters": [],
            "total_good_rasters": 0,
            "total_rasters_found": 0
        }
    }
    
    for city in cities:
        city_result = {
            "dem": None,
            "ndvi": None,
            "status": "missing"
        }
        
        dem_path = f"data/external/{city}/dem.tif"
        ndvi_path = f"data/external/{city}/green.tif"
        
        rasters_found = 0
        good_rasters = 0
        
        # Check DEM
        if os.path.exists(dem_path):
            city_result["dem"] = check_raster_metadata(dem_path)
            city_result["dem"]["path"] = dem_path
            rasters_found += 1
            if city_result["dem"].get("quality_flags", {}).get("overall_ok", False):
                good_rasters += 1
        
        # Check NDVI
        if os.path.exists(ndvi_path):
            city_result["ndvi"] = check_raster_metadata(ndvi_path)
            city_result["ndvi"]["path"] = ndvi_path
            rasters_found += 1
            if city_result["ndvi"].get("quality_flags", {}).get("overall_ok", False):
                good_rasters += 1
        
        # Determine status
        if rasters_found == 2:
            if good_rasters == 2:
                city_result["status"] = "complete_good"
                result["summary"]["cities_with_both_rasters"].append(city)
            else:
                city_result["status"] = "complete_poor"
                result["summary"]["cities_with_partial_rasters"].append(city)
        elif rasters_found == 1:
            city_result["status"] = "partial"
            result["summary"]["cities_with_partial_rasters"].append(city)
        else:
            city_result["status"] = "missing"
            result["summary"]["cities_missing_rasters"].append(city)
        
        result["cities"][city] = city_result
        result["summary"]["total_rasters_found"] += rasters_found
        result["summary"]["total_good_rasters"] += good_rasters
    
    return result


def check_config_presence() -> Dict[str, Any]:
    """Check for configuration files."""
    config_files = [
        "config/cities.yaml",
        "config/v3_overrides.yaml"
    ]
    
    result = {
        "config_files": {},
        "configs_found": [],
        "configs_missing": []
    }
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, 'r') as f:
                    content = yaml.safe_load(f)
                    result["config_files"][config_file] = {
                        "exists": True,
                        "keys": list(content.keys()) if isinstance(content, dict) else [],
                        "size": len(str(content)) if content else 0
                    }
                    result["configs_found"].append(config_file)
            except ImportError:
                result["config_files"][config_file] = {
                    "exists": True,
                    "error": "yaml module not available"
                }
                result["configs_found"].append(config_file)
            except Exception as e:
                result["config_files"][config_file] = {
                    "exists": True,
                    "error": str(e)
                }
                result["configs_found"].append(config_file)
        else:
            result["config_files"][config_file] = {"exists": False}
            result["configs_missing"].append(config_file)
    
    return result


def determine_overall_status(audit_results: Dict[str, Any]) -> str:
    """Determine PASS/PARTIAL/FAIL based on audit results."""
    # PASS: fetch script exists, CLI works, at least one city has good rasters
    has_fetch_code = (
        len(audit_results["code_presence"]["files_found"]) > 0 or
        len(audit_results["cli_help"]["successful_helps"]) > 0 or
        audit_results["imports_present"]["total_matches"] > 0
    )
    
    has_working_cli = len(audit_results["cli_help"]["successful_helps"]) > 0
    
    has_good_rasters = (
        len(audit_results["data_status"]["summary"]["cities_with_both_rasters"]) > 0
    )
    
    # PASS conditions
    if has_fetch_code and has_working_cli and has_good_rasters:
        return "PASS"
    
    # PARTIAL conditions  
    if has_fetch_code or has_working_cli:
        return "PARTIAL"
    
    # FAIL conditions
    return "FAIL"


def write_json_report(audit_results: Dict[str, Any], output_path: str) -> None:
    """Write machine-readable JSON report."""
    with open(output_path, 'w') as f:
        json.dump(audit_results, f, indent=2)


def write_markdown_report(audit_results: Dict[str, Any], output_path: str) -> None:
    """Write human-readable Markdown report."""
    overall_status = audit_results["overall_status"]
    
    with open(output_path, 'w') as f:
        f.write("# Raster Fetcher Audit Report\n\n")
        f.write(f"**Generated:** {audit_results['timestamp']}  \n")
        f.write(f"**Overall Status:** {overall_status}  \n\n")
        
        # Summary
        f.write(f"## Summary\n\n")
        if overall_status == "PASS":
            f.write("**PASS** - Working fetch capability detected with good raster data\n\n")
        elif overall_status == "PARTIAL":
            f.write("**PARTIAL** - Some fetch capability detected but incomplete data\n\n") 
        else:
            f.write("**FAIL** - No fetch capability or raster data detected\n\n")
        
        # Code Presence
        f.write("## Code Presence\n\n")
        files_found = audit_results["code_presence"]["files_found"]
        files_missing = audit_results["code_presence"]["files_missing"]
        
        if files_found:
            f.write("**Files Found:**\n")
            for file in files_found:
                f.write(f"- FOUND `{file}`\n")
            f.write("\n")
        
        if files_missing:
            f.write("**Files Missing:**\n")
            for file in files_missing:
                f.write(f"- MISSING `{file}`\n")
            f.write("\n")
        
        # Imports & Dependencies
        f.write("## Dependencies\n\n")
        deps_found = audit_results["dependencies"]["dependencies_found"]
        deps_missing = audit_results["dependencies"]["dependencies_missing"]
        
        f.write(f"**Available:** {', '.join(deps_found) if deps_found else 'None'}  \n")
        f.write(f"**Missing:** {', '.join(deps_missing) if deps_missing else 'None'}  \n\n")
        
        # CLI Help Results
        f.write("## CLI Capabilities\n\n")
        successful_helps = audit_results["cli_help"]["successful_helps"]
        failed_helps = audit_results["cli_help"]["failed_helps"]
        
        if successful_helps:
            f.write("**Working CLIs:**\n")
            for cli in successful_helps:
                f.write(f"- WORKING `{cli}`\n")
            f.write("\n")
        
        if failed_helps:
            f.write("**Failed CLIs:**\n")
            for cli in failed_helps:
                f.write(f"- FAILED `{cli}`\n")
            f.write("\n")
        
        # Data Status
        f.write("## Raster Data Status\n\n")
        data_summary = audit_results["data_status"]["summary"]
        
        f.write("| City | DEM | NDVI | Status |\n")
        f.write("|------|-----|------|--------|\n")
        
        for city, city_data in audit_results["data_status"]["cities"].items():
            dem_status = "YES" if city_data["dem"] and not city_data["dem"].get("error") else "NO"
            ndvi_status = "YES" if city_data["ndvi"] and not city_data["ndvi"].get("error") else "NO"
            
            f.write(f"| {city} | {dem_status} | {ndvi_status} | {city_data['status']} |\n")
        
        f.write(f"\n**Summary:** {data_summary['total_rasters_found']} rasters found, "
                f"{data_summary['total_good_rasters']} with good metadata\n\n")
        
        # Next Actions
        f.write("## Next Actions\n\n")
        
        if overall_status == "PASS":
            f.write("**Ready to use existing fetch capability!**\n\n")
            if successful_helps:
                f.write("**Recommended commands:**\n")
                for cli in successful_helps:
                    if cli.endswith('.py'):
                        f.write(f"```bash\npython {cli} --help\n```\n")
                    elif cli.endswith('.ps1'):
                        f.write(f"```powershell\npowershell -ExecutionPolicy Bypass -File {cli}\n```\n")
        
        elif overall_status == "PARTIAL":
            f.write("**Partial capability detected - needs completion:**\n\n")
            if deps_missing:
                f.write(f"1. Install missing dependencies: `{', '.join(deps_missing)}`\n")
            if not successful_helps:
                f.write("2. Fix or create working fetch scripts\n")
            f.write("3. Run fetch commands to populate raster data\n\n")
        
        else:
            f.write("**No fetch capability detected - needs implementation:**\n\n")
            f.write("1. Implement raster fetch scripts (STAC, auto-download)\n")
            f.write("2. Install required dependencies (rasterio, pystac-client, etc.)\n")
            f.write("3. Create AOI presets and configuration\n")
            f.write("4. Populate raster data for target cities\n\n")


def main():
    parser = argparse.ArgumentParser(description="Audit raster fetcher capabilities")
    parser.add_argument("--cities", nargs="+", default=["melbourne", "NewYork", "zurich", "dublin"],
                       help="Cities to audit")
    parser.add_argument("--year", type=int, default=2023, help="Target year")
    
    args = parser.parse_args()
    
    print("Auditing raster fetch capabilities...")
    
    # Ensure output directories exist
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/runs", exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run all audit checks
    audit_results = {
        "timestamp": datetime.now().isoformat(),
        "cities_audited": args.cities,
        "year": args.year,
        "code_presence": check_file_presence(),
        "imports_present": check_imports_in_codebase(),
        "dependencies": check_dependencies(),
        "cli_help": run_cli_dry_runs(),
        "data_status": check_data_presence(args.cities),
        "config_status": check_config_presence()
    }
    
    # Determine overall status
    audit_results["overall_status"] = determine_overall_status(audit_results)
    
    # Write reports
    json_path = f"reports/metrics/raster_fetch_audit_{timestamp}.json"
    md_path = f"reports/runs/{timestamp}_raster_fetch_audit.md"
    
    write_json_report(audit_results, json_path)
    write_markdown_report(audit_results, md_path)
    
    print(f"Audit complete. Status: {audit_results['overall_status']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()