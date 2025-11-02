#!/usr/bin/env python3
"""
A/B Model Evaluation for v2 vs v3 Environmental Features

Trains and compares CatBoost models using v2 baseline features vs v3 enhanced features
to quantify the impact of segment-based topography and seasonal canopy on prediction accuracy.

Usage:
    python pipelines/run_ab_model_eval.py melbourne 2023
    python pipelines/run_ab_model_eval.py --city melbourne --year 2023 --cv-folds 5
    python pipelines/run_ab_model_eval.py --baseline-table data/.../baseline.csv --v3-table data/.../v3.csv
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, classification_report
from sklearn.preprocessing import LabelEncoder
import catboost as cb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(table_path: str, description: str = "training data") -> Optional[pd.DataFrame]:
    """Load and validate training data CSV."""
    if not os.path.exists(table_path):
        logger.error(f"Training data not found: {table_path}")
        return None
    
    try:
        df = pd.read_csv(table_path)
        
        if df.empty:
            logger.error(f"Empty training data: {table_path}")
            return None
        
        logger.info(f"Loaded {description}: {len(df):,} rows from {os.path.basename(table_path)}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {table_path}: {e}")
        return None


def prepare_features_target(df: pd.DataFrame, target_col: str = 'volume_class',
                           exclude_patterns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare feature matrix and target vector for modeling."""
    
    if exclude_patterns is None:
        exclude_patterns = ['sensor_id', 'datetime', 'date', 'time', 'osmid', 'geometry']
    
    # Identify target column
    if target_col not in df.columns:
        # Look for common target column patterns
        possible_targets = [col for col in df.columns if any(pattern in col.lower() for pattern in 
                           ['volume', 'count', 'pedestrian', 'target', 'label', 'class'])]
        
        if possible_targets:
            target_col = possible_targets[0]
            logger.info(f"Using target column: {target_col}")
        else:
            logger.error("No suitable target column found")
            return None, None, []
    
    # Separate features and target
    feature_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        
        # Skip columns matching exclude patterns
        if any(pattern.lower() in col.lower() for pattern in exclude_patterns):
            continue
        
        # Only include numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    if not feature_cols:
        logger.error("No suitable feature columns found")
        return None, None, []
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    logger.info(f"Feature matrix: {X.shape} | Target: {len(y)} unique values: {y.nunique()}")
    logger.info(f"Features: {len(feature_cols)} columns")
    
    # Handle missing values in features
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"Handling {missing_before:,} missing values in features")
        
        # Simple median imputation for numeric features
        X = X.fillna(X.median())
        
        missing_after = X.isnull().sum().sum()
        logger.info(f"Missing values after imputation: {missing_after}")
    
    # Handle target encoding if needed
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        logger.info("Encoding string target to numeric")
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        logger.info(f"Target classes: {list(le.classes_)}")
    
    return X, y, feature_cols


def train_catboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame = None, y_val: pd.Series = None,
                        params: Dict[str, Any] = None, verbose: bool = False) -> cb.CatBoostClassifier:
    """Train CatBoost classifier with specified parameters."""
    
    if params is None:
        params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            'thread_count': -1
        }
    
    # Determine if this is a classification problem
    n_classes = len(np.unique(y_train))
    
    if n_classes == 2:
        objective = 'Logloss'
        eval_metric = 'AUC'
    elif n_classes > 2:
        objective = 'MultiClass'
        eval_metric = 'MultiClass'
    else:
        logger.error("Invalid target variable - only one class found")
        return None
    
    params.update({
        'objective': objective,
        'eval_metric': eval_metric
    })
    
    # Initialize model
    model = cb.CatBoostClassifier(**params)
    
    # Prepare evaluation set if validation data provided
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = (X_val, y_val)
    
    # Train model
    logger.info(f"Training CatBoost model with {len(X_train):,} samples, {X_train.shape[1]} features")
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=verbose,
        plot=False
    )
    
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.1f}s")
    
    return model


def evaluate_model_performance(model: cb.CatBoostClassifier, X_test: pd.DataFrame, y_test: pd.Series,
                              model_name: str = "model") -> Dict[str, Any]:
    """Evaluate trained model and return comprehensive metrics."""
    
    logger.info(f"Evaluating {model_name} performance on {len(X_test):,} test samples")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    # Macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    # Weighted averages
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    feature_names = X_test.columns.tolist()
    
    # Top 10 most important features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(10).to_dict('records')
    
    # Environmental feature importance
    env_patterns = ['grade', 'elev', 'canopy', 'topo', 'green', 'ndvi']
    env_features = importance_df[
        importance_df['feature'].str.lower().str.contains('|'.join(env_patterns), na=False)
    ].head(10).to_dict('records')
    
    metrics = {
        'model_name': model_name,
        'n_test_samples': len(X_test),
        'n_features': X_test.shape[1],
        'accuracy': float(accuracy),
        'cohen_kappa': float(kappa),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'top_features': top_features,
        'top_env_features': env_features,
        'feature_importance_stats': {
            'env_feature_count': len(env_features),
            'total_env_importance': float(sum([f['importance'] for f in env_features])),
            'max_env_importance': float(max([f['importance'] for f in env_features])) if env_features else 0.0
        }
    }
    
    # Log key metrics
    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Cohen's κ: {kappa:.4f}")
    logger.info(f"{model_name} - F1 (macro): {f1_macro:.4f}, F1 (weighted): {f1_weighted:.4f}")
    
    return metrics


def perform_cross_validation(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5,
                            params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Perform cross-validation evaluation."""
    
    logger.info(f"Performing {cv_folds}-fold cross-validation")
    
    if params is None:
        params = {
            'iterations': 500,  # Reduced for CV
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False
        }
    
    # Create model for CV
    model = cb.CatBoostClassifier(**params)
    
    # Stratified K-fold for better class balance
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    
    cv_results = {
        'cv_folds': cv_folds,
        'accuracy_scores': cv_scores.tolist(),
        'accuracy_mean': float(cv_scores.mean()),
        'accuracy_std': float(cv_scores.std()),
        'accuracy_95ci': [
            float(cv_scores.mean() - 1.96 * cv_scores.std()),
            float(cv_scores.mean() + 1.96 * cv_scores.std())
        ]
    }
    
    logger.info(f"CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    
    return cv_results


def compare_feature_importance(baseline_metrics: Dict[str, Any], v3_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Compare feature importance between baseline and v3 models."""
    
    # Extract feature importance data
    baseline_features = {f['feature']: f['importance'] for f in baseline_metrics['top_features']}
    v3_features = {f['feature']: f['importance'] for f in v3_metrics['top_features']}
    
    # Environmental feature analysis
    baseline_env_importance = baseline_metrics['feature_importance_stats']['total_env_importance']
    v3_env_importance = v3_metrics['feature_importance_stats']['total_env_importance']
    
    baseline_env_count = baseline_metrics['feature_importance_stats']['env_feature_count']
    v3_env_count = v3_metrics['feature_importance_stats']['env_feature_count']
    
    # Compare top features
    all_features = set(baseline_features.keys()) | set(v3_features.keys())
    feature_comparison = []
    
    for feature in all_features:
        baseline_imp = baseline_features.get(feature, 0)
        v3_imp = v3_features.get(feature, 0)
        
        feature_comparison.append({
            'feature': feature,
            'baseline_importance': baseline_imp,
            'v3_importance': v3_imp,
            'importance_change': v3_imp - baseline_imp,
            'relative_change': ((v3_imp - baseline_imp) / baseline_imp * 100) if baseline_imp > 0 else 0
        })
    
    # Sort by absolute change
    feature_comparison.sort(key=lambda x: abs(x['importance_change']), reverse=True)
    
    comparison = {
        'environmental_features': {
            'baseline_total_importance': baseline_env_importance,
            'v3_total_importance': v3_env_importance,
            'importance_change': v3_env_importance - baseline_env_importance,
            'baseline_feature_count': baseline_env_count,
            'v3_feature_count': v3_env_count,
            'feature_count_change': v3_env_count - baseline_env_count
        },
        'top_feature_changes': feature_comparison[:15],  # Top 15 changes
        'new_important_features': [
            f for f in v3_features.keys() 
            if f not in baseline_features and v3_features[f] > 10
        ],
        'lost_important_features': [
            f for f in baseline_features.keys()
            if f not in v3_features and baseline_features[f] > 10
        ]
    }
    
    return comparison


def run_ab_model_evaluation(city: str, year: int, baseline_table_path: str = None,
                           v3_table_path: str = None, cv_folds: int = 3) -> Dict[str, Any]:
    """Run complete A/B model evaluation comparing baseline vs v3 features."""
    
    logger.info(f"Starting A/B model evaluation for {city} ({year})")
    
    # Define default paths if not provided
    if baseline_table_path is None:
        baseline_table_path = f"data/processed/{city}/csv/feature_table_{year}_{city}_baseline.csv"
    
    if v3_table_path is None:
        v3_table_path = f"data/processed/{city}/csv/feature_table_{year}_{city}_with_env_topo_canopy.csv"
    
    # Load training data
    baseline_df = load_training_data(baseline_table_path, "baseline training data")
    v3_df = load_training_data(v3_table_path, "v3 training data")
    
    if baseline_df is None and v3_df is None:
        logger.error("No training data available for comparison")
        return {'success': False, 'error': 'No training data found'}
    
    # If only one dataset is available, create comparison with subset
    if baseline_df is None and v3_df is not None:
        logger.info("Only v3 data available - creating baseline by excluding v3-specific features")
        baseline_df = v3_df.copy()
        
        # Remove v3-specific columns for baseline comparison
        v3_patterns = ['_v3', 'spring', 'summer', 'fall', 'winter', 'evergreen', 'p85']
        baseline_cols = [col for col in baseline_df.columns 
                        if not any(pattern in col for pattern in v3_patterns)]
        baseline_df = baseline_df[baseline_cols]
        
        logger.info(f"Created baseline with {len(baseline_cols)} columns (excluded v3-specific features)")
    
    elif v3_df is None and baseline_df is not None:
        logger.warning("Only baseline data available - cannot evaluate v3 improvements")
        v3_df = baseline_df.copy()  # Fallback for comparison
    
    results = {
        'city': city,
        'year': year,
        'evaluation_timestamp': pd.Timestamp.now().isoformat(),
        'baseline_table': baseline_table_path,
        'v3_table': v3_table_path,
        'cv_folds': cv_folds
    }
    
    # Common model parameters
    model_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    }
    
    # Evaluate baseline model
    logger.info("=== Baseline Model Evaluation ===")
    
    X_baseline, y_baseline, baseline_features = prepare_features_target(baseline_df)
    
    if X_baseline is not None and y_baseline is not None:
        # Train-test split
        X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
            X_baseline, y_baseline, test_size=0.2, random_state=42, stratify=y_baseline
        )
        
        # Train baseline model
        baseline_model = train_catboost_model(X_train_base, y_train_base, params=model_params)
        
        if baseline_model:
            # Evaluate baseline
            baseline_metrics = evaluate_model_performance(
                baseline_model, X_test_base, y_test_base, "Baseline"
            )
            
            # Cross-validation for baseline
            baseline_cv = perform_cross_validation(X_baseline, y_baseline, cv_folds, model_params)
            baseline_metrics['cross_validation'] = baseline_cv
            
            results['baseline'] = baseline_metrics
            
            # Save baseline model
            models_dir = Path("data/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            baseline_model_path = models_dir / f"baseline_{city}_{year}.cbm"
            baseline_model.save_model(str(baseline_model_path))
            logger.info(f"Baseline model saved to: {baseline_model_path}")
    
    # Evaluate v3 model
    logger.info("=== v3 Enhanced Model Evaluation ===")
    
    X_v3, y_v3, v3_features = prepare_features_target(v3_df)
    
    if X_v3 is not None and y_v3 is not None:
        # Ensure same test set for fair comparison
        X_train_v3, X_test_v3, y_train_v3, y_test_v3 = train_test_split(
            X_v3, y_v3, test_size=0.2, random_state=42, stratify=y_v3
        )
        
        # Train v3 model
        v3_model = train_catboost_model(X_train_v3, y_train_v3, params=model_params)
        
        if v3_model:
            # Evaluate v3
            v3_metrics = evaluate_model_performance(
                v3_model, X_test_v3, y_test_v3, "v3 Enhanced"
            )
            
            # Cross-validation for v3
            v3_cv = perform_cross_validation(X_v3, y_v3, cv_folds, model_params)
            v3_metrics['cross_validation'] = v3_cv
            
            results['v3'] = v3_metrics
            
            # Save v3 model
            v3_model_path = models_dir / f"v3_{city}_{year}.cbm"
            v3_model.save_model(str(v3_model_path))
            logger.info(f"v3 model saved to: {v3_model_path}")
    
    # Compare models
    if 'baseline' in results and 'v3' in results:
        logger.info("=== Model Comparison ===")
        
        baseline_acc = results['baseline']['accuracy']
        v3_acc = results['v3']['accuracy']
        acc_improvement = v3_acc - baseline_acc
        
        baseline_kappa = results['baseline']['cohen_kappa']
        v3_kappa = results['v3']['cohen_kappa']
        kappa_improvement = v3_kappa - baseline_kappa
        
        # Feature importance comparison
        importance_comparison = compare_feature_importance(
            results['baseline'], results['v3']
        )
        
        comparison = {
            'performance_deltas': {
                'accuracy_baseline': baseline_acc,
                'accuracy_v3': v3_acc,
                'accuracy_improvement': acc_improvement,
                'accuracy_improvement_pp': acc_improvement * 100,  # percentage points
                'kappa_baseline': baseline_kappa,
                'kappa_v3': v3_kappa,
                'kappa_improvement': kappa_improvement,
                'kappa_improvement_pp': kappa_improvement * 100
            },
            'feature_analysis': importance_comparison,
            'statistical_significance': {
                'baseline_cv_std': results['baseline']['cross_validation']['accuracy_std'],
                'v3_cv_std': results['v3']['cross_validation']['accuracy_std'],
                'improvement_significant': abs(acc_improvement) > 2 * max(
                    results['baseline']['cross_validation']['accuracy_std'],
                    results['v3']['cross_validation']['accuracy_std']
                )
            }
        }
        
        results['comparison'] = comparison
        
        # Log comparison results
        logger.info(f"Performance Comparison:")
        logger.info(f"  Accuracy: {baseline_acc:.4f} → {v3_acc:.4f} ({acc_improvement:+.4f}, {acc_improvement*100:+.2f}pp)")
        logger.info(f"  Cohen's κ: {baseline_kappa:.4f} → {v3_kappa:.4f} ({kappa_improvement:+.4f}, {kappa_improvement*100:+.2f}pp)")
        
        env_importance_change = importance_comparison['environmental_features']['importance_change']
        logger.info(f"  Environmental feature importance change: {env_importance_change:+.1f}")
    
    # Save detailed results
    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = reports_dir / f"{city}_{year}_ab_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"A/B evaluation results saved to: {results_path}")
    
    # Save feature importance CSV
    if 'v3' in results:
        importance_dir = Path("reports/feature_importance")
        importance_dir.mkdir(parents=True, exist_ok=True)
        
        importance_df = pd.DataFrame(results['v3']['top_features'])
        importance_path = importance_dir / f"{city}_{year}_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"Feature importance saved to: {importance_path}")
    
    results['success'] = True
    return results


def main():
    """Command-line interface for A/B model evaluation."""
    parser = argparse.ArgumentParser(
        description='Run A/B model evaluation comparing baseline vs v3 features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/run_ab_model_eval.py melbourne 2023
  python pipelines/run_ab_model_eval.py --city melbourne --year 2023 --cv-folds 5
  python pipelines/run_ab_model_eval.py --baseline-table data/.../baseline.csv --v3-table data/.../v3.csv
        """)
    
    parser.add_argument('city', nargs='?', help='City name (if not using --city)')
    parser.add_argument('year', nargs='?', type=int, help='Analysis year (if not using --year)')
    
    parser.add_argument('--city', help='City name')
    parser.add_argument('--year', type=int, help='Analysis year (default: 2023)')
    
    parser.add_argument('--baseline-table', help='Path to baseline training table')
    parser.add_argument('--v3-table', help='Path to v3 enhanced training table')
    
    parser.add_argument('--cv-folds', type=int, default=3, help='Cross-validation folds (default: 3)')
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
    
    # Run evaluation
    try:
        results = run_ab_model_evaluation(
            city=city,
            year=year,
            baseline_table_path=args.baseline_table,
            v3_table_path=args.v3_table,
            cv_folds=args.cv_folds
        )
        
        if results.get('success', False):
            # Print summary
            print(f"\n=== A/B Model Evaluation Summary ===")
            print(f"City: {city.title()}, Year: {year}")
            
            if 'comparison' in results:
                comp = results['comparison']['performance_deltas']
                print(f"\nPerformance Results:")
                print(f"  Accuracy:   {comp['accuracy_baseline']:.4f} → {comp['accuracy_v3']:.4f} "
                      f"({comp['accuracy_improvement_pp']:+.2f}pp)")
                print(f"  Cohen's κ:  {comp['kappa_baseline']:.4f} → {comp['kappa_v3']:.4f} "
                      f"({comp['kappa_improvement_pp']:+.2f}pp)")
                
                env_features = results['comparison']['feature_analysis']['environmental_features']
                print(f"\nEnvironmental Features:")
                print(f"  Feature count: {env_features['baseline_feature_count']} → "
                      f"{env_features['v3_feature_count']} ({env_features['feature_count_change']:+d})")
                print(f"  Total importance: {env_features['baseline_total_importance']:.1f} → "
                      f"{env_features['v3_total_importance']:.1f} "
                      f"({env_features['importance_change']:+.1f})")
                
                is_significant = results['comparison']['statistical_significance']['improvement_significant']
                print(f"\nStatistical significance: {'Yes' if is_significant else 'No'}")
            
            print(f"\nDetailed results saved to reports/metrics/{city}_{year}_ab_metrics.json")
        
        else:
            print(f"A/B evaluation failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"A/B evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()