#!/usr/bin/env python3
"""
v2 vs v3 Environmental Feature Comparison Tool

Performs statistical comparison of v2 and v3 environmental features to validate
improvements in coverage, distribution quality, and feature characteristics.

Usage:
    python pipelines/compare_v2_v3_env.py melbourne 2023
    python pipelines/compare_v2_v3_env.py --city melbourne --year 2023 --plot
    python pipelines/compare_v2_v3_env.py --all-cities --config config/cities.yaml
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['figure.max_open_warning'] = 0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_feature_data(feature_path: str, feature_type: str) -> Optional[pd.DataFrame]:
    """Load and validate feature data CSV."""
    if not os.path.exists(feature_path):
        logger.warning(f"Feature file not found: {feature_path}")
        return None
    
    try:
        df = pd.read_csv(feature_path)
        
        if df.empty:
            logger.warning(f"Empty feature file: {feature_path}")
            return None
        
        if 'edge_osmid' not in df.columns:
            logger.warning(f"Missing edge_osmid column in: {feature_path}")
            return None
        
        logger.info(f"Loaded {feature_type}: {len(df)} edges from {os.path.basename(feature_path)}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {feature_path}: {e}")
        return None


def compute_coverage_comparison(v2_df: pd.DataFrame, v3_df: pd.DataFrame,
                               feature_columns: Dict[str, str]) -> Dict[str, Any]:
    """Compare coverage between v2 and v3 features."""
    comparison = {}
    
    # Merge on edge_osmid to get common edges
    merged = pd.merge(v2_df[['edge_osmid'] + list(feature_columns.keys())],
                     v3_df[['edge_osmid'] + list(feature_columns.values())],
                     on='edge_osmid', how='outer', suffixes=('_v2', '_v3'))
    
    total_edges = len(merged)
    
    for v2_col, v3_col in feature_columns.items():
        v2_coverage = merged[v2_col].notna().sum()
        v3_coverage = merged[v3_col].notna().sum()
        both_coverage = (merged[v2_col].notna() & merged[v3_col].notna()).sum()
        
        comparison[f"{v2_col}_vs_{v3_col}"] = {
            'v2_coverage': v2_coverage,
            'v3_coverage': v3_coverage,
            'v2_coverage_rate': v2_coverage / total_edges if total_edges > 0 else 0,
            'v3_coverage_rate': v3_coverage / total_edges if total_edges > 0 else 0,
            'both_coverage': both_coverage,
            'coverage_improvement': v3_coverage - v2_coverage,
            'coverage_improvement_rate': (v3_coverage - v2_coverage) / total_edges if total_edges > 0 else 0
        }
    
    return {
        'total_edges': total_edges,
        'feature_comparisons': comparison
    }


def compute_distribution_comparison(v2_df: pd.DataFrame, v3_df: pd.DataFrame,
                                  feature_columns: Dict[str, str]) -> Dict[str, Any]:
    """Compare statistical distributions between v2 and v3 features."""
    comparison = {}
    
    # Merge on edge_osmid to get common edges for fair comparison
    merged = pd.merge(v2_df[['edge_osmid'] + list(feature_columns.keys())],
                     v3_df[['edge_osmid'] + list(feature_columns.values())],
                     on='edge_osmid', how='inner')  # Inner join for valid comparisons
    
    for v2_col, v3_col in feature_columns.items():
        # Extract values that exist in both versions
        v2_values = merged[v2_col].dropna()
        v3_values = merged[v3_col].dropna()
        
        if len(v2_values) == 0 or len(v3_values) == 0:
            logger.warning(f"Insufficient data for comparison: {v2_col} vs {v3_col}")
            continue
        
        # Basic statistics
        v2_stats = {
            'count': len(v2_values),
            'mean': float(v2_values.mean()),
            'median': float(v2_values.median()),
            'std': float(v2_values.std()),
            'min': float(v2_values.min()),
            'max': float(v2_values.max()),
            'p25': float(v2_values.quantile(0.25)),
            'p75': float(v2_values.quantile(0.75)),
            'p90': float(v2_values.quantile(0.90)),
            'p99': float(v2_values.quantile(0.99))
        }
        
        v3_stats = {
            'count': len(v3_values),
            'mean': float(v3_values.mean()),
            'median': float(v3_values.median()),
            'std': float(v3_values.std()),
            'min': float(v3_values.min()),
            'max': float(v3_values.max()),
            'p25': float(v3_values.quantile(0.25)),
            'p75': float(v3_values.quantile(0.75)),
            'p90': float(v3_values.quantile(0.90)),
            'p99': float(v3_values.quantile(0.99))
        }
        
        # Statistical tests
        try:
            # Correlation for overlapping edges
            overlap_mask = merged[v2_col].notna() & merged[v3_col].notna()
            if overlap_mask.sum() > 10:  # Need reasonable sample size
                overlap_v2 = merged[overlap_mask][v2_col]
                overlap_v3 = merged[overlap_mask][v3_col]
                
                spearman_corr, spearman_p = stats.spearmanr(overlap_v2, overlap_v3)
                pearson_corr, pearson_p = stats.pearsonr(overlap_v2, overlap_v3)
            else:
                spearman_corr = spearman_p = pearson_corr = pearson_p = None
            
            # Kolmogorov-Smirnov test for distribution differences
            ks_stat, ks_p = stats.ks_2samp(v2_values, v3_values)
            
            # Mann-Whitney U test for median differences
            mw_stat, mw_p = stats.mannwhitneyu(v2_values, v3_values, alternative='two-sided')
            
        except Exception as e:
            logger.warning(f"Statistical test error for {v2_col} vs {v3_col}: {e}")
            spearman_corr = spearman_p = pearson_corr = pearson_p = None
            ks_stat = ks_p = mw_stat = mw_p = None
        
        comparison[f"{v2_col}_vs_{v3_col}"] = {
            'v2_stats': v2_stats,
            'v3_stats': v3_stats,
            'correlation': {
                'spearman_r': float(spearman_corr) if spearman_corr is not None else None,
                'spearman_p': float(spearman_p) if spearman_p is not None else None,
                'pearson_r': float(pearson_corr) if pearson_corr is not None else None,
                'pearson_p': float(pearson_p) if pearson_p is not None else None,
                'overlap_count': int(overlap_mask.sum()) if 'overlap_mask' in locals() else 0
            },
            'distribution_tests': {
                'ks_statistic': float(ks_stat) if ks_stat is not None else None,
                'ks_p_value': float(ks_p) if ks_p is not None else None,
                'mw_statistic': float(mw_stat) if mw_stat is not None else None,
                'mw_p_value': float(mw_p) if mw_p is not None else None
            }
        }
    
    return comparison


def create_comparison_plots(v2_df: pd.DataFrame, v3_df: pd.DataFrame,
                           feature_columns: Dict[str, str], output_dir: Path,
                           city: str, year: int) -> List[str]:
    """Create visualization plots comparing v2 and v3 features."""
    
    plot_files = []
    
    # Merge data for plotting
    merged = pd.merge(v2_df[['edge_osmid'] + list(feature_columns.keys())],
                     v3_df[['edge_osmid'] + list(feature_columns.values())],
                     on='edge_osmid', how='inner')
    
    for v2_col, v3_col in feature_columns.items():
        # Skip if insufficient data
        v2_valid = merged[v2_col].dropna()
        v3_valid = merged[v3_col].dropna()
        
        if len(v2_valid) < 10 or len(v3_valid) < 10:
            continue
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{city.title()} {year}: {v2_col} (v2) vs {v3_col} (v3)', fontsize=14)
        
        # 1. Histogram comparison
        ax1 = axes[0, 0]
        ax1.hist(v2_valid, bins=30, alpha=0.6, label='v2', density=True)
        ax1.hist(v3_valid, bins=30, alpha=0.6, label='v3', density=True)
        ax1.set_xlabel(v2_col.replace('_', ' ').title())
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution Comparison')
        ax1.legend()
        
        # 2. Scatter plot (for overlapping values)
        ax2 = axes[0, 1]
        overlap_mask = merged[v2_col].notna() & merged[v3_col].notna()
        if overlap_mask.sum() > 10:
            overlap_v2 = merged[overlap_mask][v2_col]
            overlap_v3 = merged[overlap_mask][v3_col]
            ax2.scatter(overlap_v2, overlap_v3, alpha=0.5, s=1)
            
            # Add diagonal line
            min_val = min(overlap_v2.min(), overlap_v3.min())
            max_val = max(overlap_v2.max(), overlap_v3.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Correlation info
            corr = np.corrcoef(overlap_v2, overlap_v3)[0, 1]
            ax2.set_title(f'Scatter Plot (r={corr:.3f})')
        else:
            ax2.set_title('Scatter Plot (Insufficient overlap)')
        
        ax2.set_xlabel('v2')
        ax2.set_ylabel('v3')
        
        # 3. Box plot comparison
        ax3 = axes[1, 0]
        box_data = []
        box_labels = []
        
        if len(v2_valid) > 0:
            box_data.append(v2_valid)
            box_labels.append('v2')
        
        if len(v3_valid) > 0:
            box_data.append(v3_valid)
            box_labels.append('v3')
        
        if box_data:
            ax3.boxplot(box_data, labels=box_labels)
            ax3.set_title('Box Plot Comparison')
            ax3.set_ylabel(v2_col.replace('_', ' ').title())
        
        # 4. Coverage comparison
        ax4 = axes[1, 1]
        total_edges = len(merged)
        v2_coverage = merged[v2_col].notna().sum()
        v3_coverage = merged[v3_col].notna().sum()
        
        coverage_data = [v2_coverage, v3_coverage]
        coverage_labels = ['v2', 'v3']
        
        bars = ax4.bar(coverage_labels, coverage_data)
        ax4.set_ylabel('Edges with Valid Data')
        ax4.set_title('Coverage Comparison')
        
        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, coverage_data)):
            percentage = (count / total_edges) * 100
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_edges*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        safe_col_name = v2_col.replace('/', '_').replace(' ', '_')
        plot_filename = f"env_comparison_{city}_{year}_{safe_col_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_files.append(str(plot_path))
        logger.info(f"Created comparison plot: {plot_filename}")
    
    return plot_files


def generate_comparison_report(coverage_comp: Dict[str, Any], 
                             distribution_comp: Dict[str, Any],
                             city: str, year: int, plot_files: List[str]) -> str:
    """Generate markdown report summarizing v2 vs v3 comparison."""
    
    report = f"""# Environmental Features Comparison: v2 vs v3

**City**: {city.title()}  
**Year**: {year}  
**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report compares environmental feature extraction between v2 (point-based) and v3 (segment-based topography + seasonal canopy) approaches.

### Coverage Summary

Total edges analyzed: **{coverage_comp['total_edges']:,}**

"""
    
    # Coverage improvements table
    report += "| Feature | v2 Coverage | v3 Coverage | Improvement | Improvement Rate |\n"
    report += "|---------|-------------|-------------|-------------|------------------|\n"
    
    for feature_name, stats in coverage_comp['feature_comparisons'].items():
        v2_rate = stats['v2_coverage_rate']
        v3_rate = stats['v3_coverage_rate']
        improvement = stats['coverage_improvement']
        improvement_rate = stats['coverage_improvement_rate']
        
        report += f"| {feature_name.replace('_vs_', ' → ')} | "
        report += f"{v2_rate:.1%} | {v3_rate:.1%} | "
        report += f"{improvement:+,} | {improvement_rate:+.1%} |\n"
    
    report += "\n### Key Findings\n\n"
    
    # Analyze key findings
    findings = []
    
    for feature_name, stats in coverage_comp['feature_comparisons'].items():
        improvement_rate = stats['coverage_improvement_rate']
        if improvement_rate > 0.05:  # >5% improvement
            findings.append(f"✅ **{feature_name}**: Significant coverage improvement (+{improvement_rate:.1%})")
        elif improvement_rate < -0.05:  # >5% degradation
            findings.append(f"⚠️ **{feature_name}**: Coverage decreased ({improvement_rate:.1%})")
        else:
            findings.append(f"📊 **{feature_name}**: Similar coverage ({improvement_rate:+.1%})")
    
    report += "\n".join(findings) + "\n"
    
    # Distribution analysis
    report += "\n## Statistical Analysis\n\n"
    
    for feature_name, stats in distribution_comp.items():
        v2_stats = stats['v2_stats']
        v3_stats = stats['v3_stats']
        correlation = stats['correlation']
        
        report += f"### {feature_name.replace('_vs_', ' vs ')}\n\n"
        
        # Statistics table
        report += "| Statistic | v2 | v3 | Change |\n"
        report += "|-----------|----|----|--------|\n"
        
        for stat_name in ['mean', 'median', 'std', 'p90', 'p99']:
            v2_val = v2_stats[stat_name]
            v3_val = v3_stats[stat_name]
            change = ((v3_val - v2_val) / v2_val * 100) if v2_val != 0 else 0
            
            report += f"| {stat_name.title()} | {v2_val:.3f} | {v3_val:.3f} | {change:+.1f}% |\n"
        
        # Correlation analysis
        if correlation['spearman_r'] is not None:
            spearman_r = correlation['spearman_r']
            overlap_count = correlation['overlap_count']
            
            report += f"\n**Correlation Analysis** ({overlap_count:,} overlapping edges):  \n"
            report += f"- Spearman correlation: **{spearman_r:.3f}**  \n"
            
            if spearman_r > 0.8:
                report += "- Strong positive correlation - v3 preserves v2 patterns\n"
            elif spearman_r > 0.5:
                report += "- Moderate positive correlation - v3 shows some consistency with v2\n"
            else:
                report += "- Weak correlation - v3 captures different patterns than v2\n"
        
        # Distribution tests
        dist_tests = stats['distribution_tests']
        if dist_tests['ks_p_value'] is not None:
            ks_p = dist_tests['ks_p_value']
            report += f"\n**Distribution Tests**:  \n"
            report += f"- Kolmogorov-Smirnov p-value: {ks_p:.6f}  \n"
            
            if ks_p < 0.001:
                report += "- Distributions are significantly different (p < 0.001)\n"
            elif ks_p < 0.05:
                report += "- Distributions are significantly different (p < 0.05)\n"
            else:
                report += "- No significant difference in distributions\n"
        
        report += "\n"
    
    # Plots section
    if plot_files:
        report += "## Visualization\n\n"
        report += "The following plots provide visual comparison of v2 vs v3 features:\n\n"
        
        for plot_file in plot_files:
            plot_name = Path(plot_file).name
            report += f"- `{plot_name}`\n"
        
        report += "\n"
    
    # Recommendations
    report += "## Recommendations\n\n"
    
    recommendations = []
    
    # Coverage-based recommendations
    total_coverage_improvements = sum(
        stats['coverage_improvement_rate'] 
        for stats in coverage_comp['feature_comparisons'].values()
    )
    
    if total_coverage_improvements > 0.1:  # >10% total improvement
        recommendations.append("✅ **Adopt v3**: Significant coverage improvements across features")
    elif total_coverage_improvements > 0.05:  # >5% total improvement
        recommendations.append("🔍 **Consider v3**: Moderate coverage improvements, validate model impact")
    else:
        recommendations.append("⚖️ **Evaluate carefully**: Minimal coverage changes, focus on model performance")
    
    # Correlation-based recommendations
    avg_correlation = np.mean([
        stats['correlation']['spearman_r'] 
        for stats in distribution_comp.values() 
        if stats['correlation']['spearman_r'] is not None
    ])
    
    if avg_correlation and avg_correlation > 0.8:
        recommendations.append("📊 **High consistency**: v3 features align well with v2, safe transition")
    elif avg_correlation and avg_correlation > 0.5:
        recommendations.append("📊 **Moderate consistency**: Some feature changes, monitor model performance")
    else:
        recommendations.append("⚠️ **Significant changes**: v3 features differ substantially, validate thoroughly")
    
    report += "\n".join(recommendations) + "\n"
    
    report += f"""
## Next Steps

1. **Model Evaluation**: Run A/B testing to compare model performance with v2 vs v3 features
2. **Parameter Tuning**: Consider adjusting v3 parameters if coverage is suboptimal
3. **Quality Validation**: Manually inspect extreme values and potential artifacts
4. **Production Testing**: Run v3 pipeline on a subset of production data for validation

---
*Report generated by `compare_v2_v3_env.py`*
"""
    
    return report


def compare_v2_v3_features(city: str, year: int, create_plots: bool = False) -> Dict[str, Any]:
    """Main comparison function for v2 vs v3 environmental features."""
    
    logger.info(f"Comparing v2 vs v3 environmental features for {city} ({year})")
    
    # Define file paths
    base_dir = Path.cwd()
    processed_dir = base_dir / f"data/processed/{city}/csv"
    reports_dir = base_dir / "reports/metrics"
    plots_dir = base_dir / "reports/plots"
    
    # Ensure output directories exist
    reports_dir.mkdir(parents=True, exist_ok=True)
    if create_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load v2 data
    topo_v2_path = processed_dir / f"topography_{year}_{city}.csv"
    canopy_v2_path = processed_dir / f"green_canopy_{year}_{city}.csv"
    
    topo_v2_df = load_feature_data(str(topo_v2_path), "topography v2")
    canopy_v2_df = load_feature_data(str(canopy_v2_path), "canopy v2")
    
    # Load v3 data
    topo_v3_path = processed_dir / f"topography_v3_{year}_{city}.csv"
    canopy_v3_path = processed_dir / f"green_canopy_v3_{year}_{city}.csv"
    
    topo_v3_df = load_feature_data(str(topo_v3_path), "topography v3")
    canopy_v3_df = load_feature_data(str(canopy_v3_path), "canopy v3")
    
    results = {
        'city': city,
        'year': year,
        'comparison_timestamp': pd.Timestamp.now().isoformat(),
        'data_availability': {
            'topo_v2': topo_v2_df is not None,
            'canopy_v2': canopy_v2_df is not None,
            'topo_v3': topo_v3_df is not None,
            'canopy_v3': canopy_v3_df is not None
        },
        'comparisons': {}
    }
    
    plot_files = []
    
    # Compare topography features
    if topo_v2_df is not None and topo_v3_df is not None:
        logger.info("Comparing topography features...")
        
        # Define feature column mappings (v2 -> v3)
        topo_columns = {
            'mean_grade_pct': 'grade_mean_pct_v3',
            'max_grade_pct': 'grade_max_pct_v3',
            'elev_start_m': 'elev_start_m_v3',
            'elev_end_m': 'elev_end_m_v3',
            'elev_range_m': 'elev_range_m_v3'
        }
        
        # Filter to columns that actually exist
        available_topo_columns = {
            v2_col: v3_col for v2_col, v3_col in topo_columns.items()
            if v2_col in topo_v2_df.columns and v3_col in topo_v3_df.columns
        }
        
        if available_topo_columns:
            topo_coverage = compute_coverage_comparison(topo_v2_df, topo_v3_df, available_topo_columns)
            topo_distribution = compute_distribution_comparison(topo_v2_df, topo_v3_df, available_topo_columns)
            
            results['comparisons']['topography'] = {
                'coverage': topo_coverage,
                'distribution': topo_distribution
            }
            
            if create_plots:
                topo_plots = create_comparison_plots(
                    topo_v2_df, topo_v3_df, available_topo_columns, 
                    plots_dir, city, year
                )
                plot_files.extend(topo_plots)
        else:
            logger.warning("No overlapping topography columns found for comparison")
    
    # Compare canopy features  
    if canopy_v2_df is not None and canopy_v3_df is not None:
        logger.info("Comparing canopy features...")
        
        # Define feature column mappings (v2 -> v3)
        canopy_columns = {
            'canopy_pct_buffer': 'canopy_pct_overall_v3'
        }
        
        # Add seasonal comparisons if available
        seasonal_cols = ['canopy_pct_spring_v3', 'canopy_pct_summer_v3', 
                        'canopy_pct_fall_v3', 'canopy_pct_winter_v3']
        
        for seasonal_col in seasonal_cols:
            if seasonal_col in canopy_v3_df.columns:
                # Map to v2 overall canopy for comparison
                canopy_columns[f'canopy_pct_buffer_vs_{seasonal_col}'] = seasonal_col
        
        # Filter to columns that actually exist
        available_canopy_columns = {
            v2_col: v3_col for v2_col, v3_col in canopy_columns.items()
            if v2_col in canopy_v2_df.columns and v3_col in canopy_v3_df.columns
        }
        
        if available_canopy_columns:
            canopy_coverage = compute_coverage_comparison(canopy_v2_df, canopy_v3_df, available_canopy_columns)
            canopy_distribution = compute_distribution_comparison(canopy_v2_df, canopy_v3_df, available_canopy_columns)
            
            results['comparisons']['canopy'] = {
                'coverage': canopy_coverage,
                'distribution': canopy_distribution
            }
            
            if create_plots:
                canopy_plots = create_comparison_plots(
                    canopy_v2_df, canopy_v3_df, available_canopy_columns,
                    plots_dir, city, year
                )
                plot_files.extend(canopy_plots)
        else:
            logger.warning("No overlapping canopy columns found for comparison")
    
    # Generate comparison report
    if results['comparisons']:
        logger.info("Generating comparison report...")
        
        # Combine coverage and distribution data for report
        combined_coverage = {}
        combined_distribution = {}
        
        for feature_type in ['topography', 'canopy']:
            if feature_type in results['comparisons']:
                comp = results['comparisons'][feature_type]
                if 'coverage' in comp:
                    combined_coverage.update(comp['coverage'])
                if 'distribution' in comp:
                    combined_distribution.update(comp['distribution'])
        
        report_text = generate_comparison_report(
            combined_coverage, combined_distribution, city, year, plot_files
        )
        
        # Save markdown report
        report_path = reports_dir / f"env_v2_v3_delta_{city}_{year}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Comparison report saved to: {report_path}")
        
        results['report_path'] = str(report_path)
        results['plot_files'] = plot_files
    
    # Save JSON results
    json_path = reports_dir / f"env_v2_v3_delta_{city}_{year}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Comparison data saved to: {json_path}")
    
    return results


def main():
    """Command-line interface for v2 vs v3 comparison."""
    parser = argparse.ArgumentParser(
        description='Compare v2 vs v3 environmental features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/compare_v2_v3_env.py melbourne 2023
  python pipelines/compare_v2_v3_env.py --city melbourne --year 2023 --plot
  python pipelines/compare_v2_v3_env.py --all-cities --config config/cities.yaml
        """)
    
    parser.add_argument('city', nargs='?', help='City name (if not using --city)')
    parser.add_argument('year', nargs='?', type=int, help='Analysis year (if not using --year)')
    
    parser.add_argument('--city', help='City name')
    parser.add_argument('--year', type=int, help='Analysis year (default: 2023)')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--all-cities', action='store_true', help='Compare all cities in config')
    parser.add_argument('--config', default='config/cities.yaml', help='Cities config file')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.all_cities:
        logger.info("Multi-city comparison not implemented yet - specify individual city")
        sys.exit(1)
    
    # Parse arguments
    city = args.city or args.city
    year = args.year or args.year or 2023
    
    if not city:
        parser.error("Must specify city name")
    
    # Run comparison
    try:
        results = compare_v2_v3_features(
            city=city,
            year=year,
            create_plots=args.plot
        )
        
        # Print summary
        if results['comparisons']:
            print(f"\n=== Comparison Summary for {city.title()} {year} ===")
            
            for feature_type, comp in results['comparisons'].items():
                if 'coverage' in comp:
                    coverage = comp['coverage']
                    total_edges = coverage.get('total_edges', 0)
                    print(f"\n{feature_type.title()} Features ({total_edges:,} edges):")
                    
                    for feature_name, stats in coverage.get('feature_comparisons', {}).items():
                        improvement = stats['coverage_improvement_rate']
                        v2_rate = stats['v2_coverage_rate']
                        v3_rate = stats['v3_coverage_rate']
                        
                        print(f"  {feature_name}: {v2_rate:.1%} → {v3_rate:.1%} ({improvement:+.1%})")
        
        else:
            print("No feature comparisons could be performed - check data availability")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()