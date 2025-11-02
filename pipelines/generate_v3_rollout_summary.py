#!/usr/bin/env python3
"""
v3 Rollout Summary Generator

Generates comprehensive rollout summary from v3 pipeline results across cities.
Produces both markdown report and CSV table with parameter selections,
coverage improvements, model performance deltas, and processing times.

Usage:
    python pipelines/generate_v3_rollout_summary.py --cities melbourne NewYork zurich dublin --year 2023
    python pipelines/generate_v3_rollout_summary.py --year 2023  # All cities
    python pipelines/generate_v3_rollout_summary.py --cities melbourne --detailed
"""

import argparse
import json
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_city_name(city: str) -> str:
    """Normalize city name for consistent file path resolution."""
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


def load_city_config(config_path: str = "config/cities.yaml") -> Dict[str, Any]:
    """Load cities configuration file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def get_city_parameters(config: Dict[str, Any], city: str) -> Dict[str, Any]:
    """Extract v3 parameters for a specific city."""
    defaults = config.get('defaults', {})
    cities = config.get('cities', {})
    
    # Find city config (case-insensitive)
    city_config = None
    for city_key in cities.keys():
        if city_key.lower() == city.lower():
            city_config = cities[city_key]
            break
    
    if not city_config:
        city_config = {}
    
    # Merge v3 parameters
    v3_defaults = defaults.get('v3', {})
    v3_city = city_config.get('v3', {})
    
    parameters = {
        'topo': {**v3_defaults.get('topo', {}), **v3_city.get('topo', {})},
        'canopy': {**v3_defaults.get('canopy', {}), **v3_city.get('canopy', {})}
    }
    
    # Fill in missing defaults
    topo_defaults = {
        'max_seg_m': 20,
        'smooth_window': 3,
        'dem_nodata': -32768,
        'min_valid_fraction': 0.2
    }
    
    canopy_defaults = {
        'buffer_m': 25,
        'ndvi_threshold': 0.2,
        'ndvi_rescale': 'auto',
        'treat_zero_as_nodata': True,
        'min_coverage': 0.3
    }
    
    for key, default_val in topo_defaults.items():
        if key not in parameters['topo']:
            parameters['topo'][key] = default_val
    
    for key, default_val in canopy_defaults.items():
        if key not in parameters['canopy']:
            parameters['canopy'][key] = default_val
    
    return parameters


def load_coverage_metrics(city: str, year: int) -> Dict[str, Any]:
    """Load coverage metrics from v3 pipeline results."""
    metrics = {'available': False}
    
    # Try different coverage file locations
    coverage_files = [
        f"reports/metrics/env_coverage_{city}_{year}.json",
        f"reports/metrics/env_v3_end_to_end_{city}_{year}.json"
    ]
    
    for coverage_file in coverage_files:
        if os.path.exists(coverage_file):
            try:
                with open(coverage_file, 'r') as f:
                    data = json.load(f)
                
                metrics['available'] = True
                metrics['data'] = data
                logger.debug(f"Loaded coverage metrics for {city} from {coverage_file}")
                break
                
            except Exception as e:
                logger.warning(f"Error loading coverage file {coverage_file}: {e}")
    
    return metrics


def load_comparison_metrics(city: str, year: int) -> Dict[str, Any]:
    """Load v2 vs v3 comparison metrics."""
    metrics = {'available': False}
    
    comparison_file = f"reports/metrics/env_v2_v3_delta_{city}_{year}.json"
    
    if os.path.exists(comparison_file):
        try:
            with open(comparison_file, 'r') as f:
                data = json.load(f)
            
            metrics['available'] = True
            metrics['data'] = data
            logger.debug(f"Loaded comparison metrics for {city}")
            
        except Exception as e:
            logger.warning(f"Error loading comparison file: {e}")
    
    return metrics


def load_ab_metrics(city: str, year: int) -> Dict[str, Any]:
    """Load A/B model evaluation metrics."""
    metrics = {'available': False}
    
    ab_file = f"reports/metrics/{city}_{year}_ab_metrics.json"
    
    if os.path.exists(ab_file):
        try:
            with open(ab_file, 'r') as f:
                data = json.load(f)
            
            metrics['available'] = True
            metrics['data'] = data
            logger.debug(f"Loaded A/B metrics for {city}")
            
        except Exception as e:
            logger.warning(f"Error loading A/B metrics file: {e}")
    
    return metrics


def load_timing_metrics(city: str, year: int) -> Dict[str, Any]:
    """Load timing metrics from pipeline execution."""
    timing = {'available': False, 'stages': {}, 'total': 0}
    
    # Look for timing files
    timing_pattern = f"reports/metrics/timing_*_{city}_{year}.json"
    timing_files = list(Path("reports/metrics").glob(f"timing_*_{city}_{year}.json"))
    
    if timing_files:
        timing['available'] = True
        
        for timing_file in timing_files:
            try:
                with open(timing_file, 'r') as f:
                    data = json.load(f)
                
                stage = data.get('stage', timing_file.stem)
                duration = data.get('duration_seconds', data.get('seconds', 0))
                success = data.get('success', data.get('rc', 0) == 0)
                
                timing['stages'][stage] = {
                    'duration': duration,
                    'success': success
                }
                
                timing['total'] += duration
                
            except Exception as e:
                logger.warning(f"Error loading timing file {timing_file}: {e}")
    
    # Fallback: check coverage metrics for timing info
    if not timing['available']:
        coverage_metrics = load_coverage_metrics(city, year)
        if coverage_metrics['available'] and 'timing' in coverage_metrics['data']:
            timing_data = coverage_metrics['data']['timing']
            timing['available'] = True
            timing['stages'] = timing_data
            timing['total'] = sum(timing_data.values())
    
    return timing


def compute_coverage_improvements(comparison_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract coverage improvements from comparison data."""
    improvements = {
        'topography': {'available': False},
        'canopy': {'available': False}
    }
    
    if not comparison_metrics['available']:
        return improvements
    
    data = comparison_metrics['data']
    
    # Extract from comparison data structure
    if 'comparisons' in data:
        comparisons = data['comparisons']
        
        # Topography improvements
        if 'topography' in comparisons and 'coverage' in comparisons['topography']:
            topo_coverage = comparisons['topography']['coverage']
            if 'feature_comparisons' in topo_coverage:
                improvements['topography']['available'] = True
                improvements['topography']['details'] = {}
                
                for feature_name, stats in topo_coverage['feature_comparisons'].items():
                    improvements['topography']['details'][feature_name] = {
                        'v2_coverage': stats.get('v2_coverage_rate', 0),
                        'v3_coverage': stats.get('v3_coverage_rate', 0),
                        'improvement': stats.get('coverage_improvement_rate', 0),
                        'absolute_improvement': stats.get('coverage_improvement', 0)
                    }
        
        # Canopy improvements
        if 'canopy' in comparisons and 'coverage' in comparisons['canopy']:
            canopy_coverage = comparisons['canopy']['coverage']
            if 'feature_comparisons' in canopy_coverage:
                improvements['canopy']['available'] = True
                improvements['canopy']['details'] = {}
                
                for feature_name, stats in canopy_coverage['feature_comparisons'].items():
                    improvements['canopy']['details'][feature_name] = {
                        'v2_coverage': stats.get('v2_coverage_rate', 0),
                        'v3_coverage': stats.get('v3_coverage_rate', 0),
                        'improvement': stats.get('coverage_improvement_rate', 0),
                        'absolute_improvement': stats.get('coverage_improvement', 0)
                    }
    
    return improvements


def extract_model_deltas(ab_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model performance deltas from A/B evaluation."""
    deltas = {'available': False}
    
    if not ab_metrics['available']:
        return deltas
    
    data = ab_metrics['data']
    
    if 'comparison' in data and 'performance_deltas' in data['comparison']:
        perf_deltas = data['comparison']['performance_deltas']
        
        deltas['available'] = True
        deltas['accuracy_baseline'] = perf_deltas.get('accuracy_baseline', 0)
        deltas['accuracy_v3'] = perf_deltas.get('accuracy_v3', 0)
        deltas['accuracy_improvement'] = perf_deltas.get('accuracy_improvement', 0)
        deltas['accuracy_improvement_pp'] = perf_deltas.get('accuracy_improvement_pp', 0)
        
        deltas['kappa_baseline'] = perf_deltas.get('kappa_baseline', 0)
        deltas['kappa_v3'] = perf_deltas.get('kappa_v3', 0)
        deltas['kappa_improvement'] = perf_deltas.get('kappa_improvement', 0)
        deltas['kappa_improvement_pp'] = perf_deltas.get('kappa_improvement_pp', 0)
        
        # Statistical significance
        if 'statistical_significance' in data['comparison']:
            sig_data = data['comparison']['statistical_significance']
            deltas['improvement_significant'] = sig_data.get('improvement_significant', False)
        
        # Feature importance changes
        if 'feature_analysis' in data['comparison']:
            feat_analysis = data['comparison']['feature_analysis']
            if 'environmental_features' in feat_analysis:
                env_features = feat_analysis['environmental_features']
                deltas['env_feature_importance_change'] = env_features.get('importance_change', 0)
                deltas['env_feature_count_change'] = env_features.get('feature_count_change', 0)
    
    return deltas


def identify_next_actions(city: str, coverage_metrics: Dict[str, Any], 
                         ab_metrics: Dict[str, Any], timing: Dict[str, Any]) -> List[str]:
    """Identify next actions and flags for the city."""
    actions = []
    flags = []
    
    # Check timing flags
    total_time_minutes = timing['total'] / 60 if timing['total'] > 0 else 0
    if total_time_minutes > 45:
        flags.append(f"⏰ Long processing time: {total_time_minutes:.1f} min")
        actions.append("Consider parameter optimization or hardware scaling")
    
    # Check coverage issues
    if coverage_metrics['available']:
        data = coverage_metrics['data']
        if 'coverage' in data:
            # Check topography coverage
            if 'topography' in data['coverage']:
                topo_cov = data['coverage']['topography'].get('coverage', 0)
                if topo_cov < 0.3:
                    flags.append("⚠️ Low topography coverage")
                    actions.append("Check DEM data quality and coverage")
            
            # Check canopy coverage
            if 'canopy' in data['coverage']:
                canopy_cov = data['coverage']['canopy'].get('coverage', 0)
                if canopy_cov < 0.2:
                    flags.append("⚠️ Low canopy coverage")
                    actions.append("Consider vector canopy fallback or improve NDVI data")
                
                # Check for seasonal data
                canopy_columns = data['coverage']['canopy'].get('columns_found', [])
                seasonal_cols = [col for col in canopy_columns if any(season in col for season in ['spring', 'summer', 'fall', 'winter'])]
                
                if len(seasonal_cols) == 0:
                    flags.append("📅 No seasonal NDVI data")
                    actions.append("Verify multi-band NDVI availability or enable vector fallback")
    
    # Check model performance issues
    if ab_metrics['available']:
        deltas = extract_model_deltas(ab_metrics)
        if deltas['available']:
            acc_improvement = deltas.get('accuracy_improvement_pp', 0)
            kappa_improvement = deltas.get('kappa_improvement_pp', 0)
            
            if acc_improvement < -1:  # > 1pp regression
                flags.append(f"📉 Model accuracy regression: {acc_improvement:.1f}pp")
                actions.append("Review v3 features; consider fallback to v2")
            
            if kappa_improvement < -1:  # > 1pp regression
                flags.append(f"📉 Model kappa regression: {kappa_improvement:.1f}pp")
                actions.append("Investigate feature quality issues")
            
            if acc_improvement < 0.5 and kappa_improvement < 0.5:
                actions.append("Consider parameter tuning: increase buffer_m or adjust ndvi_threshold")
    
    # Generic recommendations
    if not coverage_metrics['available']:
        flags.append("❓ Coverage data missing")
        actions.append("Re-run v3 pipeline to generate coverage metrics")
    
    if not ab_metrics['available']:
        flags.append("❓ Model evaluation missing")
        actions.append("Run A/B model evaluation to assess performance impact")
    
    # Combine flags and actions
    all_actions = flags + [f"→ {action}" for action in actions]
    
    if not all_actions:
        all_actions.append("✅ No issues detected - ready for production")
    
    return all_actions


def generate_city_summary_row(city: str, year: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary data for a single city."""
    logger.info(f"Processing summary for {city}")
    
    # Get parameters
    parameters = get_city_parameters(config, city)
    
    # Load metrics
    coverage_metrics = load_coverage_metrics(city, year)
    comparison_metrics = load_comparison_metrics(city, year)
    ab_metrics = load_ab_metrics(city, year)
    timing = load_timing_metrics(city, year)
    
    # Extract data
    coverage_improvements = compute_coverage_improvements(comparison_metrics)
    model_deltas = extract_model_deltas(ab_metrics)
    next_actions = identify_next_actions(city, coverage_metrics, ab_metrics, timing)
    
    # Build summary row
    row = {
        'city': city,
        'year': year,
        'normalized_city': normalize_city_name(city),
        
        # Parameters
        'max_seg_m': parameters['topo'].get('max_seg_m', 'N/A'),
        'smooth_window': parameters['topo'].get('smooth_window', 'N/A'),
        'buffer_m': parameters['canopy'].get('buffer_m', 'N/A'),
        'ndvi_threshold': parameters['canopy'].get('ndvi_threshold', 'N/A'),
        
        # Coverage improvements
        'topo_coverage_improvement': 'N/A',
        'canopy_coverage_improvement': 'N/A',
        
        # Model performance
        'accuracy_baseline': 'N/A',
        'accuracy_v3': 'N/A',
        'accuracy_improvement_pp': 'N/A',
        'kappa_baseline': 'N/A',
        'kappa_v3': 'N/A',
        'kappa_improvement_pp': 'N/A',
        'improvement_significant': 'N/A',
        
        # Timing
        'total_time_minutes': timing['total'] / 60 if timing['total'] > 0 else 'N/A',
        'env_v3_time': timing['stages'].get('env_v3', {}).get('duration', 'N/A'),
        'compare_time': timing['stages'].get('compare_v2_v3', {}).get('duration', 'N/A'),
        'ab_eval_time': timing['stages'].get('ab_eval', {}).get('duration', 'N/A'),
        
        # Flags and actions
        'next_actions': next_actions,
        'flags_count': len([action for action in next_actions if not action.startswith('→')]),
        
        # Availability flags
        'coverage_data_available': coverage_metrics['available'],
        'comparison_data_available': comparison_metrics['available'],
        'ab_data_available': ab_metrics['available'],
        'timing_data_available': timing['available']
    }
    
    # Fill in coverage improvements if available
    if coverage_improvements['topography']['available']:
        topo_details = coverage_improvements['topography']['details']
        # Use mean grade as primary metric
        if any('mean_grade' in key for key in topo_details.keys()):
            grade_key = next(key for key in topo_details.keys() if 'mean_grade' in key)
            improvement = topo_details[grade_key]['improvement']
            row['topo_coverage_improvement'] = f"{improvement:.1%}"
    
    if coverage_improvements['canopy']['available']:
        canopy_details = coverage_improvements['canopy']['details']
        # Use overall canopy as primary metric
        if any('overall' in key for key in canopy_details.keys()):
            canopy_key = next(key for key in canopy_details.keys() if 'overall' in key)
            improvement = canopy_details[canopy_key]['improvement']
            row['canopy_coverage_improvement'] = f"{improvement:.1%}"
    
    # Fill in model deltas if available
    if model_deltas['available']:
        row['accuracy_baseline'] = f"{model_deltas['accuracy_baseline']:.4f}"
        row['accuracy_v3'] = f"{model_deltas['accuracy_v3']:.4f}"
        row['accuracy_improvement_pp'] = f"{model_deltas['accuracy_improvement_pp']:+.2f}pp"
        row['kappa_baseline'] = f"{model_deltas['kappa_baseline']:.4f}"
        row['kappa_v3'] = f"{model_deltas['kappa_v3']:.4f}"
        row['kappa_improvement_pp'] = f"{model_deltas['kappa_improvement_pp']:+.2f}pp"
        row['improvement_significant'] = model_deltas.get('improvement_significant', False)
    
    return row


def generate_markdown_report(summary_data: List[Dict[str, Any]], year: int) -> str:
    """Generate comprehensive markdown rollout summary."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# v3 Rollout Summary Report

**Generated**: {timestamp}  
**Year**: {year}  
**Cities Processed**: {len(summary_data)}  

## Executive Summary

This report summarizes the v3 environmental features rollout across {len(summary_data)} cities, including parameter selections, coverage improvements, model performance deltas, and processing times.

"""
    
    # Count successful cities
    successful_cities = len([city for city in summary_data if city['coverage_data_available']])
    cities_with_improvements = len([city for city in summary_data 
                                  if city['topo_coverage_improvement'] != 'N/A' or 
                                     city['canopy_coverage_improvement'] != 'N/A'])
    cities_with_model_eval = len([city for city in summary_data if city['ab_data_available']])
    
    report += f"""### Overview
- **Successfully Processed**: {successful_cities}/{len(summary_data)} cities
- **Coverage Analysis Available**: {cities_with_improvements} cities  
- **Model Evaluation Available**: {cities_with_model_eval} cities
- **Average Processing Time**: {sum([city['total_time_minutes'] for city in summary_data if isinstance(city['total_time_minutes'], (int, float))]) / max(1, len([city for city in summary_data if isinstance(city['total_time_minutes'], (int, float))]))::.1f} minutes

"""
    
    # Parameter Selection per City
    report += """## Parameter Selection per City

The following parameters were selected for each city based on terrain characteristics and data availability:

| City | Max Segment (m) | Smooth Window | Buffer (m) | NDVI Threshold |
|------|-----------------|---------------|------------|----------------|
"""
    
    for city in summary_data:
        report += f"| {city['city'].title()} | {city['max_seg_m']} | {city['smooth_window']} | {city['buffer_m']} | {city['ndvi_threshold']} |\n"
    
    # Coverage Improvements
    report += f"""
## Coverage Improvements (v2 → v3)

Feature coverage improvements from segment-based topography and seasonal canopy processing:

| City | Topography | Canopy | Status |
|------|------------|--------|--------|
"""
    
    for city in summary_data:
        topo_imp = city['topo_coverage_improvement']
        canopy_imp = city['canopy_coverage_improvement']
        
        # Status indicators
        if topo_imp == 'N/A' and canopy_imp == 'N/A':
            status = '❓ No comparison data'
        elif (topo_imp != 'N/A' and float(topo_imp.rstrip('%')) > 5) or (canopy_imp != 'N/A' and float(canopy_imp.rstrip('%')) > 5):
            status = '✅ Significant improvement'
        elif (topo_imp != 'N/A' and float(topo_imp.rstrip('%')) > 0) or (canopy_imp != 'N/A' and float(canopy_imp.rstrip('%')) > 0):
            status = '📈 Moderate improvement'
        else:
            status = '📊 Similar coverage'
        
        report += f"| {city['city'].title()} | {topo_imp} | {canopy_imp} | {status} |\n"
    
    # Model Performance Deltas
    report += f"""
## Model Performance Deltas

A/B evaluation results comparing baseline vs v3-enhanced models:

| City | Accuracy Δ | Cohen's κ Δ | Significance | Status |
|------|------------|-------------|--------------|--------|
"""
    
    for city in summary_data:
        acc_delta = city['accuracy_improvement_pp']
        kappa_delta = city['kappa_improvement_pp']
        significant = city['improvement_significant']
        
        if acc_delta == 'N/A':
            status = '❓ No evaluation data'
        elif significant and (acc_delta.startswith('+') or kappa_delta.startswith('+')):
            status = '🎯 Significant improvement'
        elif acc_delta.startswith('-') and float(acc_delta.rstrip('pp').replace('+', '')) < -1:
            status = '⚠️ Regression detected'
        elif acc_delta.startswith('+') or kappa_delta.startswith('+'):
            status = '📈 Improvement'
        else:
            status = '📊 Similar performance'
        
        sig_indicator = '✓' if significant == True else '○' if significant == False else 'N/A'
        
        report += f"| {city['city'].title()} | {acc_delta} | {kappa_delta} | {sig_indicator} | {status} |\n"
    
    # Processing Times
    report += f"""
## Processing Times

Processing duration per city and stage:

| City | Total Time | v3 Extraction | Comparison | A/B Eval | Status |
|------|------------|---------------|------------|----------|--------|
"""
    
    for city in summary_data:
        total_time = city['total_time_minutes']
        env_time = city['env_v3_time']
        compare_time = city['compare_time']
        ab_time = city['ab_eval_time']
        
        if isinstance(total_time, (int, float)):
            if total_time > 45:
                status = '⏰ Slow processing'
            elif total_time > 30:
                status = '🕐 Moderate time'
            else:
                status = '⚡ Fast processing'
            total_str = f"{total_time:.1f} min"
        else:
            status = '❓ No timing data'
            total_str = str(total_time)
        
        # Format individual times
        env_str = f"{env_time:.1f}s" if isinstance(env_time, (int, float)) else str(env_time)
        comp_str = f"{compare_time:.1f}s" if isinstance(compare_time, (int, float)) else str(compare_time)
        ab_str = f"{ab_time:.1f}s" if isinstance(ab_time, (int, float)) else str(ab_time)
        
        report += f"| {city['city'].title()} | {total_str} | {env_str} | {comp_str} | {ab_str} | {status} |\n"
    
    # Next Actions and Flags
    report += f"""
## Next Actions & Recommendations

City-specific recommendations and flags identified during rollout:

"""
    
    for city in summary_data:
        city_actions = city['next_actions']
        flags_count = city['flags_count']
        
        report += f"""### {city['city'].title()}
"""
        
        if flags_count > 0:
            report += f"**Issues Identified**: {flags_count}\n\n"
        
        for action in city_actions:
            report += f"- {action}\n"
        
        report += "\n"
    
    # Global Recommendations
    report += """## Global Recommendations

Based on the rollout results across all cities:

"""
    
    # Analyze global patterns
    global_recommendations = []
    
    # Check processing times
    cities_with_slow_processing = [city for city in summary_data 
                                 if isinstance(city['total_time_minutes'], (int, float)) and city['total_time_minutes'] > 45]
    if len(cities_with_slow_processing) > 0:
        global_recommendations.append(f"⏰ **Performance**: {len(cities_with_slow_processing)} cities have processing times >45min. Consider parameter optimization or infrastructure scaling.")
    
    # Check model improvements
    cities_with_improvements = [city for city in summary_data 
                              if city['accuracy_improvement_pp'] != 'N/A' and city['accuracy_improvement_pp'].startswith('+')]
    cities_with_regressions = [city for city in summary_data 
                             if city['accuracy_improvement_pp'] != 'N/A' and city['accuracy_improvement_pp'].startswith('-') 
                             and float(city['accuracy_improvement_pp'].rstrip('pp').replace('+', '')) < -1]
    
    if len(cities_with_improvements) > len(cities_with_regressions):
        global_recommendations.append(f"✅ **Deployment Ready**: {len(cities_with_improvements)} cities show model improvements vs {len(cities_with_regressions)} with regressions. Recommend proceeding with v3 rollout.")
    elif len(cities_with_regressions) > 0:
        global_recommendations.append(f"⚠️ **Caution**: {len(cities_with_regressions)} cities show model regressions. Review parameter settings and data quality before full deployment.")
    
    # Check coverage improvements
    coverage_available = len([city for city in summary_data if city['comparison_data_available']])
    if coverage_available < len(summary_data) / 2:
        global_recommendations.append(f"📊 **Data Gaps**: Only {coverage_available}/{len(summary_data)} cities have comparison data. Run v2 vs v3 comparison for remaining cities.")
    
    if not global_recommendations:
        global_recommendations.append("🎉 **Success**: No global issues detected. v3 rollout appears successful across all cities.")
    
    for recommendation in global_recommendations:
        report += f"- {recommendation}\n"
    
    report += f"""
## Summary Statistics

- **Total Cities Processed**: {len(summary_data)}
- **Average Processing Time**: {sum([city['total_time_minutes'] for city in summary_data if isinstance(city['total_time_minutes'], (int, float))]) / max(1, len([city for city in summary_data if isinstance(city['total_time_minutes'], (int, float))]))::.1f} minutes
- **Cities with Coverage Improvements**: {cities_with_improvements}/{len(summary_data)}
- **Cities with Model Evaluation**: {cities_with_model_eval}/{len(summary_data)}
- **Cities Flagged for Issues**: {len([city for city in summary_data if city['flags_count'] > 0])}/{len(summary_data)}

---
*Generated by `generate_v3_rollout_summary.py` on {timestamp}*
"""
    
    return report


def generate_csv_table(summary_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate CSV summary table."""
    
    # Prepare rows for CSV
    csv_rows = []
    
    for city in summary_data:
        row = {
            'City': city['city'],
            'Year': city['year'],
            'Max_Seg_m': city['max_seg_m'],
            'Smooth_Window': city['smooth_window'], 
            'Buffer_m': city['buffer_m'],
            'NDVI_Threshold': city['ndvi_threshold'],
            'Topo_Coverage_Improvement': city['topo_coverage_improvement'],
            'Canopy_Coverage_Improvement': city['canopy_coverage_improvement'],
            'Accuracy_Baseline': city['accuracy_baseline'],
            'Accuracy_v3': city['accuracy_v3'],
            'Accuracy_Improvement_pp': city['accuracy_improvement_pp'],
            'Kappa_Baseline': city['kappa_baseline'],
            'Kappa_v3': city['kappa_v3'],
            'Kappa_Improvement_pp': city['kappa_improvement_pp'],
            'Improvement_Significant': city['improvement_significant'],
            'Total_Time_Minutes': city['total_time_minutes'],
            'Env_v3_Time_s': city['env_v3_time'],
            'Compare_Time_s': city['compare_time'],
            'AB_Eval_Time_s': city['ab_eval_time'],
            'Flags_Count': city['flags_count'],
            'Coverage_Data_Available': city['coverage_data_available'],
            'Comparison_Data_Available': city['comparison_data_available'],
            'AB_Data_Available': city['ab_data_available'],
            'Timing_Data_Available': city['timing_data_available']
        }
        
        csv_rows.append(row)
    
    df = pd.DataFrame(csv_rows)
    return df


def generate_rollout_summary(cities: List[str], year: int = 2023, 
                           config_path: str = "config/cities.yaml") -> Dict[str, Any]:
    """Generate complete rollout summary for specified cities."""
    
    logger.info(f"Generating v3 rollout summary for {len(cities)} cities ({year})")
    
    # Load configuration
    config = load_city_config(config_path)
    
    # Process each city
    summary_data = []
    
    for city in cities:
        try:
            city_summary = generate_city_summary_row(city, year, config)
            summary_data.append(city_summary)
            
        except Exception as e:
            logger.error(f"Error processing {city}: {e}")
            # Add minimal entry for failed city
            summary_data.append({
                'city': city,
                'year': year,
                'error': str(e),
                'next_actions': [f"❌ Processing failed: {e}"]
            })
    
    # Generate outputs
    markdown_report = generate_markdown_report(summary_data, year)
    csv_table = generate_csv_table(summary_data)
    
    # Save outputs
    reports_dir = Path("reports/runs")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save markdown report
    markdown_path = reports_dir / f"{timestamp}_v3_rollout_summary.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    # Save CSV table
    csv_path = reports_dir / f"{timestamp}_v3_rollout_table.csv"
    csv_table.to_csv(csv_path, index=False)
    
    # Save raw data
    json_path = reports_dir / f"{timestamp}_v3_rollout_data.json"
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"Rollout summary generated:")
    logger.info(f"  Markdown: {markdown_path}")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Data: {json_path}")
    
    return {
        'summary_data': summary_data,
        'markdown_report': markdown_report,
        'csv_table': csv_table,
        'files': {
            'markdown': str(markdown_path),
            'csv': str(csv_path),
            'json': str(json_path)
        }
    }


def print_console_summary(summary_data: List[Dict[str, Any]]):
    """Print concise summary to console."""
    
    print(f"\n{'='*80}")
    print(f"v3 ROLLOUT SUMMARY - {len(summary_data)} CITIES")
    print(f"{'='*80}")
    
    print(f"{'City':<12} {'Status':<8} {'Time':<8} {'Coverage':<12} {'Model Δ':<12} {'Flags':<6}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*6}")
    
    for city in summary_data:
        city_name = city['city'][:11]
        
        # Status
        if city.get('error'):
            status = 'FAILED'
        elif city['coverage_data_available']:
            status = 'SUCCESS'
        else:
            status = 'PARTIAL'
        
        # Time
        time_val = city['total_time_minutes']
        time_str = f"{time_val:.1f}m" if isinstance(time_val, (int, float)) else "N/A"
        
        # Coverage (show topo/canopy)
        topo_imp = city['topo_coverage_improvement']
        canopy_imp = city['canopy_coverage_improvement']
        
        if topo_imp != 'N/A' and canopy_imp != 'N/A':
            coverage_str = f"{topo_imp}/{canopy_imp}"
        elif topo_imp != 'N/A':
            coverage_str = f"{topo_imp}/N/A"
        elif canopy_imp != 'N/A':
            coverage_str = f"N/A/{canopy_imp}"
        else:
            coverage_str = "N/A"
        
        coverage_str = coverage_str[:11]
        
        # Model delta (show accuracy)
        acc_delta = city['accuracy_improvement_pp']
        model_str = acc_delta[:11] if acc_delta != 'N/A' else "N/A"
        
        # Flags
        flags_count = city['flags_count']
        flags_str = str(flags_count) if isinstance(flags_count, int) else "0"
        
        print(f"{city_name:<12} {status:<8} {time_str:<8} {coverage_str:<12} {model_str:<12} {flags_str:<6}")
    
    # Summary statistics
    successful = len([city for city in summary_data if city.get('coverage_data_available', False)])
    with_improvements = len([city for city in summary_data if city['topo_coverage_improvement'] != 'N/A' or city['canopy_coverage_improvement'] != 'N/A'])
    with_model_eval = len([city for city in summary_data if city['ab_data_available']])
    avg_time = sum([city['total_time_minutes'] for city in summary_data if isinstance(city['total_time_minutes'], (int, float))]) / max(1, len([city for city in summary_data if isinstance(city['total_time_minutes'], (int, float))]))
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {successful}/{len(summary_data)} successful | {with_improvements} with coverage data | {with_model_eval} with model eval | {avg_time:.1f}m avg time")
    print(f"{'='*80}\n")


def main():
    """Command-line interface for rollout summary generation."""
    parser = argparse.ArgumentParser(
        description='Generate v3 rollout summary from pipeline results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/generate_v3_rollout_summary.py --cities melbourne NewYork zurich dublin --year 2023
  python pipelines/generate_v3_rollout_summary.py --year 2023  # All cities
  python pipelines/generate_v3_rollout_summary.py --cities melbourne --detailed
        """)
    
    parser.add_argument('--cities', nargs='+', 
                       help='Cities to include (default: all cities in config)')
    parser.add_argument('--year', type=int, default=2023, help='Analysis year')
    parser.add_argument('--config', default='config/cities.yaml', 
                       help='Configuration file path')
    parser.add_argument('--detailed', action='store_true', 
                       help='Include detailed per-city analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine cities to process
    if args.cities:
        cities = args.cities
    else:
        # Load all cities from config
        config = load_city_config(args.config)
        cities = list(config.get('cities', {}).keys())
        
        if not cities:
            cities = ['melbourne', 'NewYork', 'zurich', 'dublin']  # Default fallback
    
    logger.info(f"Processing cities: {cities}")
    
    # Generate summary
    try:
        result = generate_rollout_summary(cities, args.year, args.config)
        
        # Print console summary
        print_console_summary(result['summary_data'])
        
        # Print file locations
        print(f"📄 Detailed report: {result['files']['markdown']}")
        print(f"📊 CSV table: {result['files']['csv']}")
        print(f"📋 Raw data: {result['files']['json']}")
        
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()