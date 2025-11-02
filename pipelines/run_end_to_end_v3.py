#!/usr/bin/env python3
"""
End-to-End v3 Pipeline Runner (Single City)

Orchestrates complete v3 environmental feature extraction and integration
for a single city, including sensor pipeline integration and validation.

Usage:
    python pipelines/run_end_to_end_v3.py melbourne 2023
    python pipelines/run_end_to_end_v3.py --city NewYork --year 2023
    python pipelines/run_end_to_end_v3.py zurich --config config/cities.yaml
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_city_name(city: str) -> str:
    """Normalize city name for consistent file path resolution."""
    # Handle common variations
    name_mappings = {
        'newyork': 'NewYork',
        'new_york': 'NewYork', 
        'new york': 'NewYork',
        'melbourne': 'melbourne',
        'zurich': 'zurich',
        'dublin': 'dublin'
    }
    
    city_lower = city.lower().replace(' ', '').replace('_', '')
    return name_mappings.get(city_lower, city)


def load_city_config(config_path: str, city: str) -> Dict[str, Any]:
    """Load configuration for a specific city."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        defaults = config.get('defaults', {})
        cities = config.get('cities', {})
        
        # Find city config (case-insensitive)
        city_config = None
        for city_key in cities.keys():
            if city_key.lower() == city.lower():
                city_config = cities[city_key]
                break
        
        if not city_config:
            logger.warning(f"City '{city}' not found in config, using defaults")
            city_config = {}
        
        # Merge defaults with city-specific config
        merged_config = {}
        
        # Merge v3 parameters
        v3_defaults = defaults.get('v3', {})
        v3_city = city_config.get('v3', {})
        
        merged_config['v3'] = {}
        for category in ['topo', 'canopy']:
            merged_config['v3'][category] = {**v3_defaults.get(category, {}), 
                                           **v3_city.get(category, {})}
        
        # Other config
        merged_config['emit_v2_names'] = defaults.get('emit_v2_names', True)
        merged_config['year'] = city_config.get('default_year', defaults.get('year', 2023))
        merged_config['paths'] = city_config.get('paths', {})
        
        logger.info(f"Loaded config for {city}")
        return merged_config
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def check_required_files(city: str, year: int) -> Tuple[bool, list]:
    """Check if required input files exist for the city."""
    normalized_city = normalize_city_name(city)
    
    required_files = [
        f"data/processed/{normalized_city}/gpkg/street_network_{year}_{city}.gpkg",
        f"data/processed/{normalized_city}/raster/dem_{year}_{city}.tif",
        f"data/processed/{normalized_city}/raster/green_{year}_{city}.tif"
    ]
    
    # Alternative file patterns to check
    alternatives = [
        f"data/processed/{city}/gpkg/street_network_{year}_{city}.gpkg",
        f"data/processed/{city}/raster/dem_{year}_{city}.tif", 
        f"data/processed/{city}/raster/green_{year}_{city}.tif"
    ]
    
    missing_files = []
    for req_file, alt_file in zip(required_files, alternatives):
        if not (os.path.exists(req_file) or os.path.exists(alt_file)):
            missing_files.append(req_file)
    
    return len(missing_files) == 0, missing_files


def run_command_with_logging(cmd: list, description: str, timeout: int = 3600) -> Dict[str, Any]:
    """Execute a command and return execution details."""
    logger.info(f"Running {description}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        if success:
            logger.info(f"✓ {description} completed in {duration:.1f}s")
        else:
            logger.error(f"✗ {description} failed (rc={result.returncode})")
            if result.stderr:
                logger.error(f"Error output: {result.stderr.strip()}")
        
        return {
            'success': success,
            'duration': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'command': cmd
        }
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"✗ {description} timed out after {timeout}s")
        return {
            'success': False,
            'duration': duration,
            'return_code': -1,
            'error': f"Command timed out after {timeout}s",
            'command': cmd
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ {description} failed with exception: {e}")
        return {
            'success': False,
            'duration': duration,
            'return_code': -1,
            'error': str(e),
            'command': cmd
        }


def compute_file_coverage_stats(file_path: str, feature_columns: list) -> Dict[str, Any]:
    """Compute basic coverage statistics for a feature file."""
    if not os.path.exists(file_path):
        return {
            'file_exists': False,
            'total_edges': 0,
            'coverage': 0.0,
            'columns_found': []
        }
    
    try:
        df = pd.read_csv(file_path)
        
        if df.empty or 'edge_osmid' not in df.columns:
            return {
                'file_exists': True,
                'total_edges': 0,
                'coverage': 0.0,
                'columns_found': []
            }
        
        total_edges = len(df)
        columns_found = [col for col in feature_columns if col in df.columns]
        
        if columns_found:
            # Use first available column for coverage
            primary_col = columns_found[0]
            non_null_count = df[primary_col].notna().sum()
            coverage = non_null_count / total_edges if total_edges > 0 else 0.0
        else:
            coverage = 0.0
        
        return {
            'file_exists': True,
            'total_edges': total_edges,
            'coverage': coverage,
            'columns_found': columns_found,
            'feature_columns_coverage': {
                col: df[col].notna().sum() for col in columns_found
            }
        }
        
    except Exception as e:
        logger.error(f"Error computing coverage for {file_path}: {e}")
        return {
            'file_exists': True,
            'total_edges': 0,
            'coverage': 0.0,
            'error': str(e),
            'columns_found': []
        }


def run_v3_environmental_extraction(city: str, year: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run v3 environmental feature extraction."""
    logger.info("=== Step 1: v3 Environmental Feature Extraction ===")
    
    # Build command for run_env_v3.py
    cmd = [
        sys.executable, "pipelines/run_env_v3.py",
        "--city", city,
        "--year", str(year)
    ]
    
    if config.get('emit_v2_names', True):
        cmd.append("--force-v3")  # Ensure v3 extraction with v2 compatibility names
    
    # Add verbose flag for debugging
    cmd.append("--verbose")
    
    result = run_command_with_logging(cmd, "v3 environmental extraction", timeout=1800)  # 30 min
    
    # Check if output files were created
    normalized_city = normalize_city_name(city)
    topo_file = f"data/processed/{normalized_city}/csv/topography_v3_{year}_{city}.csv"
    canopy_file = f"data/processed/{normalized_city}/csv/green_canopy_v3_{year}_{city}.csv"
    
    # Check alternative paths
    if not os.path.exists(topo_file):
        alt_topo = f"data/processed/{city}/csv/topography_v3_{year}_{city}.csv"
        if os.path.exists(alt_topo):
            topo_file = alt_topo
    
    if not os.path.exists(canopy_file):
        alt_canopy = f"data/processed/{city}/csv/green_canopy_v3_{year}_{city}.csv"
        if os.path.exists(alt_canopy):
            canopy_file = alt_canopy
    
    # Compute coverage statistics
    topo_coverage = compute_file_coverage_stats(
        topo_file, 
        ['grade_mean_pct_v3', 'grade_p85_pct_v3', 'grade_max_pct_v3']
    )
    
    canopy_coverage = compute_file_coverage_stats(
        canopy_file,
        ['canopy_pct_overall_v3', 'canopy_pct_spring_v3', 'canopy_pct_summer_v3']
    )
    
    result.update({
        'output_files': {
            'topography': topo_file,
            'canopy': canopy_file
        },
        'coverage': {
            'topography': topo_coverage,
            'canopy': canopy_coverage
        }
    })
    
    return result


def run_sensor_pipeline_integration(city: str, year: int) -> Dict[str, Any]:
    """Run sensor-edge mapping and environmental aggregation."""
    logger.info("=== Step 2: Sensor Pipeline Integration ===")
    
    # Look for existing sensor pipeline scripts
    sensor_scripts = [
        "scripts/run_sensor_env_pipeline.ps1",
        "pipelines/run_sensor_env_pipeline.ps1", 
        "src/data_processing/run_sensor_pipeline.py"
    ]
    
    sensor_script = None
    for script in sensor_scripts:
        if os.path.exists(script):
            sensor_script = script
            break
    
    if not sensor_script:
        logger.warning("No sensor pipeline script found - skipping sensor integration")
        return {
            'success': False,
            'skipped': True,
            'reason': 'No sensor pipeline script found',
            'duration': 0
        }
    
    # Execute sensor pipeline
    if sensor_script.endswith('.ps1'):
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", sensor_script, city, str(year)]
    else:
        cmd = [sys.executable, sensor_script, city, str(year)]
    
    result = run_command_with_logging(cmd, "sensor pipeline integration", timeout=900)  # 15 min
    
    return result


def run_v2_v3_comparison(city: str, year: int) -> Dict[str, Any]:
    """Run comparison between v2 and v3 features if v2 exists."""
    logger.info("=== Step 3: v2 vs v3 Comparison ===")
    
    # Check if v2 files exist
    normalized_city = normalize_city_name(city)
    v2_topo = f"data/processed/{normalized_city}/csv/topography_{year}_{city}.csv"
    v2_canopy = f"data/processed/{normalized_city}/csv/green_canopy_{year}_{city}.csv"
    
    # Check alternative paths
    if not os.path.exists(v2_topo):
        alt_v2_topo = f"data/processed/{city}/csv/topography_{year}_{city}.csv"
        if os.path.exists(alt_v2_topo):
            v2_topo = alt_v2_topo
    
    if not os.path.exists(v2_canopy):
        alt_v2_canopy = f"data/processed/{city}/csv/green_canopy_{year}_{city}.csv"
        if os.path.exists(alt_v2_canopy):
            v2_canopy = alt_v2_canopy
    
    if not (os.path.exists(v2_topo) or os.path.exists(v2_canopy)):
        logger.info("No v2 features found - skipping comparison")
        return {
            'success': False,
            'skipped': True,
            'reason': 'No v2 features available for comparison',
            'duration': 0
        }
    
    # Run comparison
    cmd = [
        sys.executable, "pipelines/compare_v2_v3_env.py",
        "--city", city,
        "--year", str(year),
        "--verbose"
    ]
    
    result = run_command_with_logging(cmd, "v2 vs v3 comparison", timeout=600)  # 10 min
    
    return result


def run_ab_model_evaluation(city: str, year: int) -> Dict[str, Any]:
    """Run A/B model evaluation comparing baseline vs v3-enhanced models."""
    logger.info("=== Step 4: A/B Model Evaluation ===")
    
    # Check if training data exists
    normalized_city = normalize_city_name(city)
    training_file = f"data/processed/{normalized_city}/csv/feature_table_{year}_{city}_with_env_topo_canopy.csv"
    
    if not os.path.exists(training_file):
        alt_training = f"data/processed/{city}/csv/feature_table_{year}_{city}_with_env_topo_canopy.csv"
        if os.path.exists(alt_training):
            training_file = alt_training
    
    if not os.path.exists(training_file):
        logger.warning(f"Training data not found at {training_file} - skipping A/B evaluation")
        return {
            'success': False,
            'skipped': True,
            'reason': 'No training data available',
            'duration': 0
        }
    
    # Run A/B model evaluation
    cmd = [
        sys.executable, "pipelines/run_ab_model_eval.py",
        "--city", city,
        "--year", str(year),
        "--cv-folds", "3",
        "--verbose"
    ]
    
    result = run_command_with_logging(cmd, "A/B model evaluation", timeout=1800)  # 30 min
    
    return result


def generate_coverage_summary(city: str, year: int, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive coverage summary."""
    normalized_city = normalize_city_name(city)
    
    summary = {
        'city': city,
        'year': year,
        'normalized_city': normalized_city,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'pipeline_success': True,
        'steps': {},
        'coverage': {},
        'timing': {}
    }
    
    # Process pipeline step results
    for step_name, step_result in pipeline_results.items():
        summary['steps'][step_name] = {
            'success': step_result.get('success', False),
            'duration': step_result.get('duration', 0),
            'skipped': step_result.get('skipped', False),
            'reason': step_result.get('reason', '')
        }
        
        summary['timing'][step_name] = step_result.get('duration', 0)
        
        if not step_result.get('success', False) and not step_result.get('skipped', False):
            summary['pipeline_success'] = False
    
    # Extract coverage information
    if 'v3_extraction' in pipeline_results and 'coverage' in pipeline_results['v3_extraction']:
        summary['coverage'] = pipeline_results['v3_extraction']['coverage']
    
    # Total processing time
    summary['total_duration'] = sum(summary['timing'].values())
    
    # Determine overall success
    critical_steps = ['v3_extraction']
    summary['pipeline_success'] = all(
        pipeline_results.get(step, {}).get('success', False) 
        for step in critical_steps
    )
    
    return summary


def save_coverage_report(summary: Dict[str, Any], city: str, year: int):
    """Save coverage report to standardized location."""
    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = reports_dir / f"env_coverage_{city}_{year}.json"
    
    try:
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Coverage report saved: {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to save coverage report: {e}")


def run_end_to_end_v3_pipeline(city: str, year: int, config_path: str = "config/cities.yaml") -> bool:
    """Run complete end-to-end v3 pipeline for a single city."""
    
    start_time = time.time()
    logger.info(f"Starting end-to-end v3 pipeline for {city} ({year})")
    
    # Load configuration
    config = load_city_config(config_path, city)
    
    # Check required files
    files_exist, missing_files = check_required_files(city, year)
    if not files_exist:
        logger.error(f"Missing required input files for {city}:")
        for file in missing_files:
            logger.error(f"  - {file}")
        return False
    
    # Initialize pipeline results
    pipeline_results = {}
    
    try:
        # Step 1: v3 Environmental Extraction
        pipeline_results['v3_extraction'] = run_v3_environmental_extraction(city, year, config)
        
        # Step 2: Sensor Pipeline Integration (optional)
        pipeline_results['sensor_integration'] = run_sensor_pipeline_integration(city, year)
        
        # Step 3: v2 vs v3 Comparison (optional) 
        pipeline_results['v2_v3_comparison'] = run_v2_v3_comparison(city, year)
        
        # Step 4: A/B Model Evaluation (optional)
        pipeline_results['ab_evaluation'] = run_ab_model_evaluation(city, year)
        
        # Generate and save coverage summary
        coverage_summary = generate_coverage_summary(city, year, pipeline_results)
        save_coverage_report(coverage_summary, city, year)
        
        # Final status
        total_time = time.time() - start_time
        success = coverage_summary['pipeline_success']
        
        logger.info(f"Pipeline completed in {total_time:.1f}s - Success: {success}")
        
        # Print summary
        print(f"\n=== {city.upper()} {year} PIPELINE SUMMARY ===")
        print(f"Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"Duration: {total_time:.1f}s")
        
        for step_name, step_info in coverage_summary['steps'].items():
            status = '✓' if step_info['success'] else '✗' if not step_info['skipped'] else '○'
            duration = step_info['duration']
            reason = f" ({step_info['reason']})" if step_info.get('reason') else ""
            print(f"  {status} {step_name}: {duration:.1f}s{reason}")
        
        if 'coverage' in coverage_summary:
            topo_cov = coverage_summary['coverage'].get('topography', {}).get('coverage', 0)
            canopy_cov = coverage_summary['coverage'].get('canopy', {}).get('coverage', 0)
            print(f"Coverage: Topo {topo_cov:.1%}, Canopy {canopy_cov:.1%}")
        
        return success
        
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        return False


def main():
    """Command-line interface for end-to-end v3 pipeline."""
    parser = argparse.ArgumentParser(
        description='Run end-to-end v3 environmental feature pipeline for a single city',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/run_end_to_end_v3.py melbourne 2023
  python pipelines/run_end_to_end_v3.py --city NewYork --year 2023
  python pipelines/run_end_to_end_v3.py zurich --config config/cities.yaml
        """)
    
    parser.add_argument('city', nargs='?', help='City name (if not using --city)')
    parser.add_argument('year', nargs='?', type=int, help='Analysis year (if not using --year)')
    
    parser.add_argument('--city', help='City name')
    parser.add_argument('--year', type=int, help='Analysis year (default: 2023)')
    parser.add_argument('--config', default='config/cities.yaml', help='Configuration file path')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse arguments
    city = args.city or args.city
    year = args.year or args.year or 2023
    
    if not city:
        parser.error("Must specify city name")
    
    # Run pipeline
    success = run_end_to_end_v3_pipeline(city, year, args.config)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()