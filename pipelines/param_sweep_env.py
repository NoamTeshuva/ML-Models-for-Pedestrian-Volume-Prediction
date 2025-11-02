#!/usr/bin/env python3
"""
Environmental Feature Parameter Sweep Harness

Tests different parameter combinations for v3 topography and green canopy extraction
to optimize coverage and feature quality across cities.

Usage:
    python pipelines/param_sweep_env.py melbourne 2023 --feature topography
    python pipelines/param_sweep_env.py melbourne 2023 --feature canopy --quick
    python pipelines/param_sweep_env.py --config sweep_configs/melbourne.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, Any, List, Tuple

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


def get_topography_parameter_grid(sweep_type: str = 'default') -> Dict[str, List]:
    """Define parameter grid for topography feature extraction."""
    
    if sweep_type == 'quick':
        return {
            'max_seg_m': [15, 20, 30],
            'smooth_window': [3, 5],
            'min_valid_fraction': [0.2, 0.3]
        }
    elif sweep_type == 'comprehensive':
        return {
            'max_seg_m': [10, 15, 20, 25, 30, 40],
            'smooth_window': [1, 3, 5, 7],
            'min_valid_fraction': [0.1, 0.2, 0.3, 0.4]
        }
    else:  # default
        return {
            'max_seg_m': [15, 20, 25],
            'smooth_window': [3, 5],
            'min_valid_fraction': [0.2, 0.3]
        }


def get_canopy_parameter_grid(sweep_type: str = 'default') -> Dict[str, List]:
    """Define parameter grid for green canopy feature extraction."""
    
    if sweep_type == 'quick':
        return {
            'buffer_m': [20, 25, 30],
            'ndvi_threshold': [0.2, 0.3],
            'min_coverage': [0.3, 0.4]
        }
    elif sweep_type == 'comprehensive':
        return {
            'buffer_m': [15, 20, 25, 30, 40, 50],
            'ndvi_threshold': [0.1, 0.2, 0.25, 0.3, 0.4],
            'min_coverage': [0.2, 0.3, 0.4, 0.5],
            'treat_zero_as_nodata': [True, False]
        }
    else:  # default
        return {
            'buffer_m': [20, 25, 30],
            'ndvi_threshold': [0.2, 0.25, 0.3],
            'min_coverage': [0.3, 0.4]
        }


def compute_feature_quality_metrics(df: pd.DataFrame, feature_type: str) -> Dict[str, float]:
    """Compute quality metrics for extracted features."""
    if df.empty:
        return {'coverage': 0.0, 'completeness': 0.0, 'validity': 0.0}
    
    total_edges = len(df)
    
    if feature_type == 'topography':
        # Key topography metrics
        primary_cols = ['grade_mean_pct_v3', 'grade_max_pct_v3', 'elev_range_m_v3']
        
        # Coverage: fraction of edges with non-null primary features
        non_null_count = df[primary_cols[0]].notna().sum() if primary_cols[0] in df.columns else 0
        coverage = non_null_count / total_edges if total_edges > 0 else 0.0
        
        # Completeness: fraction of edges with all topography features
        if all(col in df.columns for col in primary_cols):
            complete_count = df[primary_cols].notna().all(axis=1).sum()
            completeness = complete_count / total_edges
        else:
            completeness = 0.0
        
        # Validity: fraction with reasonable grade values (0-50%)
        if 'grade_max_pct_v3' in df.columns:
            valid_grades = ((df['grade_max_pct_v3'] >= 0) & 
                           (df['grade_max_pct_v3'] <= 50)).sum()
            validity = valid_grades / total_edges
        else:
            validity = 0.0
            
        # Additional metrics
        mean_valid_seg_frac = (df['valid_seg_frac_v3'].mean() 
                              if 'valid_seg_frac_v3' in df.columns else 0.0)
        
        return {
            'coverage': coverage,
            'completeness': completeness, 
            'validity': validity,
            'mean_valid_seg_fraction': mean_valid_seg_frac,
            'total_edges': total_edges
        }
        
    elif feature_type == 'canopy':
        # Key canopy metrics
        primary_cols = ['canopy_pct_overall_v3']
        seasonal_cols = ['canopy_pct_spring_v3', 'canopy_pct_summer_v3', 
                        'canopy_pct_fall_v3', 'canopy_pct_winter_v3']
        
        # Coverage: fraction of edges with overall canopy values
        non_null_count = df[primary_cols[0]].notna().sum() if primary_cols[0] in df.columns else 0
        coverage = non_null_count / total_edges if total_edges > 0 else 0.0
        
        # Seasonal completeness: fraction with all seasonal bands
        seasonal_available = all(col in df.columns for col in seasonal_cols)
        if seasonal_available:
            seasonal_complete = df[seasonal_cols].notna().all(axis=1).sum()
            seasonal_completeness = seasonal_complete / total_edges
        else:
            seasonal_completeness = 0.0
        
        # Validity: values in [0, 1] range
        if 'canopy_pct_overall_v3' in df.columns:
            valid_values = ((df['canopy_pct_overall_v3'] >= 0) & 
                           (df['canopy_pct_overall_v3'] <= 1)).sum()
            validity = valid_values / total_edges
        else:
            validity = 0.0
        
        # Vector fallback usage
        vector_fallback_used = 0
        if 'canopy_pct_vector_v3' in df.columns:
            vector_fallback_used = df['canopy_pct_vector_v3'].notna().sum()
        
        return {
            'coverage': coverage,
            'seasonal_completeness': seasonal_completeness,
            'validity': validity,
            'vector_fallback_used': vector_fallback_used,
            'vector_fallback_rate': vector_fallback_used / total_edges if total_edges > 0 else 0.0,
            'seasonal_available': seasonal_available,
            'total_edges': total_edges
        }
    
    return {'coverage': 0.0, 'completeness': 0.0, 'validity': 0.0}


def run_topography_parameter_sweep(network_path: str, dem_path: str, temp_dir: Path,
                                  param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Run parameter sweep for topography features."""
    results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Running topography sweep with {len(combinations)} parameter combinations")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        # Add default values for missing parameters
        params.setdefault('dem_nodata', -32768)
        
        logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")
        
        # Generate unique output path
        output_path = temp_dir / f"topo_sweep_{i:03d}.csv"
        
        try:
            start_time = time.time()
            
            # Run extraction
            df = extract_topography_features_v3(
                network_path=network_path,
                dem_path=dem_path,
                output_path=str(output_path),
                max_seg_m=params['max_seg_m'],
                smooth_window=params['smooth_window'],
                min_valid_fraction=params['min_valid_fraction'],
                dem_nodata=params['dem_nodata'],
                emit_v2_names=False
            )
            
            processing_time = time.time() - start_time
            
            # Compute quality metrics
            quality_metrics = compute_feature_quality_metrics(df, 'topography')
            
            # Store results
            result = {
                'combination_id': i,
                'parameters': params,
                'processing_time': processing_time,
                'success': True,
                'quality_metrics': quality_metrics
            }
            results.append(result)
            
            logger.info(f"  → Coverage: {quality_metrics['coverage']:.1%}, "
                       f"Time: {processing_time:.1f}s")
            
            # Clean up temp file
            if output_path.exists():
                output_path.unlink()
                
        except Exception as e:
            logger.error(f"  → Failed: {e}")
            result = {
                'combination_id': i,
                'parameters': params,
                'processing_time': 0.0,
                'success': False,
                'error': str(e),
                'quality_metrics': {'coverage': 0.0, 'completeness': 0.0, 'validity': 0.0}
            }
            results.append(result)
    
    return results


def run_canopy_parameter_sweep(network_path: str, ndvi_path: str, temp_dir: Path,
                              param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Run parameter sweep for green canopy features."""
    results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Running canopy sweep with {len(combinations)} parameter combinations")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        # Add default values for missing parameters
        params.setdefault('ndvi_rescale', 'auto')
        params.setdefault('treat_zero_as_nodata', True)
        
        logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")
        
        # Generate unique output path
        output_path = temp_dir / f"canopy_sweep_{i:03d}.csv"
        
        try:
            start_time = time.time()
            
            # Run extraction
            df = extract_green_canopy_features_v3(
                network_path=network_path,
                ndvi_path=ndvi_path,
                output_path=str(output_path),
                buffer_m=params['buffer_m'],
                ndvi_threshold=params['ndvi_threshold'],
                ndvi_rescale=params['ndvi_rescale'],
                min_coverage=params['min_coverage'],
                treat_zero_as_nodata=params['treat_zero_as_nodata'],
                emit_v2_names=False
            )
            
            processing_time = time.time() - start_time
            
            # Compute quality metrics
            quality_metrics = compute_feature_quality_metrics(df, 'canopy')
            
            # Store results
            result = {
                'combination_id': i,
                'parameters': params,
                'processing_time': processing_time,
                'success': True,
                'quality_metrics': quality_metrics
            }
            results.append(result)
            
            logger.info(f"  → Coverage: {quality_metrics['coverage']:.1%}, "
                       f"Seasonal: {quality_metrics['seasonal_completeness']:.1%}, "
                       f"Time: {processing_time:.1f}s")
            
            # Clean up temp file
            if output_path.exists():
                output_path.unlink()
                
        except Exception as e:
            logger.error(f"  → Failed: {e}")
            result = {
                'combination_id': i,
                'parameters': params,
                'processing_time': 0.0,
                'success': False,
                'error': str(e),
                'quality_metrics': {'coverage': 0.0, 'seasonal_completeness': 0.0, 'validity': 0.0}
            }
            results.append(result)
    
    return results


def analyze_sweep_results(results: List[Dict[str, Any]], feature_type: str) -> Dict[str, Any]:
    """Analyze parameter sweep results and identify optimal parameters."""
    if not results:
        return {'error': 'No results to analyze'}
    
    # Convert to DataFrame for easier analysis
    rows = []
    for result in results:
        row = {'combination_id': result['combination_id'], 'success': result['success']}
        row.update(result['parameters'])
        row.update(result.get('quality_metrics', {}))
        row['processing_time'] = result.get('processing_time', 0)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if df.empty or df['success'].sum() == 0:
        return {'error': 'No successful parameter combinations'}
    
    # Focus on successful runs
    success_df = df[df['success']].copy()
    
    # Define scoring function based on feature type
    if feature_type == 'topography':
        # Score based on coverage, completeness, and validity
        success_df['score'] = (
            0.4 * success_df['coverage'] +
            0.3 * success_df['completeness'] + 
            0.2 * success_df['validity'] +
            0.1 * success_df['mean_valid_seg_fraction']
        )
    else:  # canopy
        # Score based on coverage, seasonal completeness, and validity
        success_df['score'] = (
            0.4 * success_df['coverage'] +
            0.3 * success_df['seasonal_completeness'] +
            0.2 * success_df['validity'] +
            0.1 * (1 - success_df['vector_fallback_rate'])  # Prefer less fallback
        )
    
    # Find best parameter combination
    best_idx = success_df['score'].idxmax()
    best_params = success_df.loc[best_idx]
    
    # Parameter sensitivity analysis
    param_cols = [col for col in success_df.columns 
                 if col not in ['combination_id', 'success', 'score', 'processing_time'] 
                 and col not in ['coverage', 'completeness', 'validity', 
                               'mean_valid_seg_fraction', 'seasonal_completeness',
                               'vector_fallback_used', 'vector_fallback_rate', 
                               'seasonal_available', 'total_edges']]
    
    param_sensitivity = {}
    for param in param_cols:
        if success_df[param].nunique() > 1:  # Only analyze varying parameters
            corr = success_df[param].corr(success_df['score'])
            param_sensitivity[param] = {
                'correlation_with_score': corr,
                'values_tested': sorted(success_df[param].unique().tolist()),
                'best_value': best_params[param]
            }
    
    return {
        'total_combinations_tested': len(results),
        'successful_combinations': len(success_df),
        'best_score': best_params['score'],
        'best_parameters': {param: best_params[param] for param in param_cols},
        'best_quality_metrics': {
            col: best_params[col] for col in best_params.index 
            if col in ['coverage', 'completeness', 'validity', 
                      'mean_valid_seg_fraction', 'seasonal_completeness',
                      'vector_fallback_rate']
        },
        'parameter_sensitivity': param_sensitivity,
        'processing_time_stats': {
            'mean': success_df['processing_time'].mean(),
            'median': success_df['processing_time'].median(),
            'best_params_time': best_params['processing_time']
        }
    }


def run_parameter_sweep(city: str, year: int, feature_type: str, sweep_type: str = 'default',
                       config_path: Optional[str] = None) -> Dict[str, Any]:
    """Run parameter sweep for specified feature type."""
    
    logger.info(f"Starting {feature_type} parameter sweep for {city} ({year})")
    
    # Define file paths
    network_path = f"data/processed/{city}/gpkg/street_network_{year}_{city}.gpkg"
    dem_path = f"data/processed/{city}/raster/dem_{year}_{city}.tif"
    ndvi_path = f"data/processed/{city}/raster/green_{year}_{city}.tif"
    
    # Check for multi-band NDVI
    ndvi_multiband = f"data/processed/{city}/raster/green_{year}_{city}_multiband.tif"
    if os.path.exists(ndvi_multiband):
        ndvi_path = ndvi_multiband
        logger.info(f"Using multi-band NDVI: {ndvi_multiband}")
    
    # Create temp directory
    temp_dir = Path(f"temp/param_sweep_{city}_{year}_{feature_type}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if feature_type == 'topography':
            # Check required files
            if not os.path.exists(network_path):
                raise FileNotFoundError(f"Network file not found: {network_path}")
            if not os.path.exists(dem_path):
                raise FileNotFoundError(f"DEM file not found: {dem_path}")
            
            # Get parameter grid
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                param_grid = config.get('topography_parameters', 
                                      get_topography_parameter_grid(sweep_type))
            else:
                param_grid = get_topography_parameter_grid(sweep_type)
            
            # Run sweep
            results = run_topography_parameter_sweep(network_path, dem_path, temp_dir, param_grid)
            
        elif feature_type == 'canopy':
            # Check required files
            if not os.path.exists(network_path):
                raise FileNotFoundError(f"Network file not found: {network_path}")
            if not os.path.exists(ndvi_path):
                raise FileNotFoundError(f"NDVI file not found: {ndvi_path}")
            
            # Get parameter grid
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                param_grid = config.get('canopy_parameters', 
                                      get_canopy_parameter_grid(sweep_type))
            else:
                param_grid = get_canopy_parameter_grid(sweep_type)
            
            # Run sweep
            results = run_canopy_parameter_sweep(network_path, ndvi_path, temp_dir, param_grid)
            
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Analyze results
        analysis = analyze_sweep_results(results, feature_type)
        
        return {
            'city': city,
            'year': year,
            'feature_type': feature_type,
            'sweep_type': sweep_type,
            'timestamp': time.time(),
            'results': results,
            'analysis': analysis
        }
        
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            for file in temp_dir.glob('*'):
                try:
                    file.unlink()
                except Exception:
                    pass
            try:
                temp_dir.rmdir()
            except Exception:
                pass


def main():
    """Command-line interface for parameter sweep."""
    parser = argparse.ArgumentParser(
        description='Run parameter sweep for v3 environmental features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/param_sweep_env.py melbourne 2023 --feature topography
  python pipelines/param_sweep_env.py melbourne 2023 --feature canopy --quick  
  python pipelines/param_sweep_env.py --config sweep_configs/melbourne.yaml
        """)
    
    parser.add_argument('city', nargs='?', help='City name')
    parser.add_argument('year', nargs='?', type=int, help='Analysis year')
    
    parser.add_argument('--feature', choices=['topography', 'canopy'], 
                       default='topography', help='Feature type to sweep')
    parser.add_argument('--sweep-type', choices=['quick', 'default', 'comprehensive'],
                       default='default', help='Scope of parameter sweep')
    
    parser.add_argument('--config', help='YAML config file with parameter grids')
    parser.add_argument('--output', help='Output file for results (default: reports/metrics/)')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.config:
        if not args.city or not args.year:
            parser.error("Must specify city and year, or provide --config file")
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        city = config.get('city', args.city)
        year = config.get('year', args.year)
        feature_type = config.get('feature_type', args.feature)
        sweep_type = config.get('sweep_type', args.sweep_type)
    else:
        city = args.city
        year = args.year
        feature_type = args.feature
        sweep_type = args.sweep_type
    
    # Run parameter sweep
    results = run_parameter_sweep(
        city=city,
        year=year,
        feature_type=feature_type,
        sweep_type=sweep_type,
        config_path=args.config
    )
    
    # Save results
    output_dir = Path("reports/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"env_param_sweep_{city}_{year}_{feature_type}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    analysis = results['analysis']
    if 'error' in analysis:
        logger.error(f"Parameter sweep failed: {analysis['error']}")
        sys.exit(1)
    
    logger.info(f"\nParameter sweep completed:")
    logger.info(f"  Tested: {analysis['total_combinations_tested']} combinations")
    logger.info(f"  Successful: {analysis['successful_combinations']} combinations")
    logger.info(f"  Best score: {analysis['best_score']:.3f}")
    logger.info(f"  Best parameters: {analysis['best_parameters']}")
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()