#!/usr/bin/env python3
"""
Training Pipeline Orchestrator

Loads merged feature tables, trains CatBoost models, and evaluates performance
with cross-city validation and environmental feature ablation studies.

Usage:
    python pipelines/run_training.py --cities melbourne new_york --features baseline+env
    python pipelines/run_training.py --config config/cities.yaml --experiment loco_validation
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import LeaveOneGroupOut, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class TrainingOrchestrator:
    def __init__(self, config_path: str = "config/cities.yaml"):
        """Initialize training orchestrator with configuration."""
        self.config = self.load_config(config_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Create output directories
        os.makedirs("reports/metrics", exist_ok=True)
        os.makedirs("reports/feature_importance", exist_ok=True)
        os.makedirs("reports/runs", exist_ok=True)
        os.makedirs("data/models", exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def find_feature_tables(self, cities: List[str], year: Optional[int] = None) -> Dict[str, str]:
        """Find feature table paths for specified cities."""
        feature_tables = {}
        
        for city in cities:
            city_config = self.config['cities'].get(city, {})
            target_year = year or city_config.get('default_year', self.config['defaults']['year'])
            
            # Try multiple path patterns
            possible_paths = [
                f"data/processed/{city}/csv/feature_table_{target_year}_{city}_with_env_topo_canopy.csv",
                f"data/processed/{city.title()}/csv/feature_table_{target_year}_{city}_with_env_topo_canopy.csv",
                f"data/processed/{city}/csv/feature_table_{target_year}_{city}_aligned.csv"
            ]
            
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path:
                feature_tables[city] = found_path
                print(f"Found feature table for {city}: {found_path}")
            else:
                print(f"WARNING: No feature table found for {city}, tried: {possible_paths}")
        
        return feature_tables
    
    def load_and_merge_data(self, feature_tables: Dict[str, str]) -> pd.DataFrame:
        """Load and merge feature tables from multiple cities."""
        dfs = []
        
        for city, path in feature_tables.items():
            print(f"Loading data for {city}...")
            
            try:
                df = pd.read_csv(path)
                df['city'] = city  # Add city identifier
                
                # Standardize column names
                df = self.standardize_columns(df, city)
                
                # Basic validation
                required_cols = ['sensor_id', 'date', 'time', 'count']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"WARNING: Missing columns in {city}: {missing_cols}")
                    continue
                
                print(f"Loaded {len(df):,} records for {city}")
                dfs.append(df)
                
            except Exception as e:
                print(f"ERROR loading data for {city}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid feature tables loaded")
        
        # Merge all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        print(f"Total merged dataset: {len(merged_df):,} records from {len(dfs)} cities")
        
        return merged_df
    
    def standardize_columns(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """Standardize column names based on city configuration."""
        city_config = self.config['cities'].get(city, {})
        sensor_schema = city_config.get('sensor_schema', {})
        
        # Column mapping
        column_mapping = {
            sensor_schema.get('id_column', 'sensor_id'): 'sensor_id',
            sensor_schema.get('name_column', 'sensor_name'): 'sensor_name',
            sensor_schema.get('lat_column', 'latitude'): 'latitude',
            sensor_schema.get('lon_column', 'longitude'): 'longitude',
            sensor_schema.get('date_column', 'date'): 'date',
            sensor_schema.get('time_column', 'time'): 'time',
            sensor_schema.get('count_column', 'count'): 'count'
        }
        
        # Apply mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df = df.rename(columns={old_col: new_col})
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, feature_groups: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix based on specified feature groups."""
        
        # Get feature group definitions
        all_feature_groups = self.config['training']['feature_groups']
        
        selected_features = []
        for group in feature_groups:
            if group in all_feature_groups:
                selected_features.extend(all_feature_groups[group])
            else:
                print(f"WARNING: Unknown feature group '{group}'")
        
        # Add temporal features if not present
        df = self.add_temporal_features(df)
        
        # Filter to available features
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = [f for f in selected_features if f not in df.columns]
        
        if missing_features:
            print(f"Missing features: {missing_features}")
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Create feature matrix
        feature_df = df[['sensor_id', 'city', 'date', 'time', 'count'] + available_features].copy()
        
        # Handle missing values
        feature_df = self.handle_missing_values(feature_df, available_features)
        
        return feature_df, available_features
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from date/time columns."""
        
        # Combine date and time
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # Extract temporal features
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['datetime'].dt.dayofweek
        if 'month' not in df.columns:
            df['month'] = df['datetime'].dt.month
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['day_of_week'] >= 5
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Handle missing values in feature columns."""
        
        env_features = ['mean_grade_pct', 'max_grade_pct', 'elev_start_m', 
                       'elev_end_m', 'elev_range_m', 'canopy_pct_buffer']
        
        for col in feature_columns:
            if col in df.columns:
                missing_rate = df[col].isna().mean()
                if missing_rate > 0:
                    print(f"Column {col}: {missing_rate:.1%} missing values")
                    
                    if col in env_features:
                        # Use median imputation for environmental features
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                    else:
                        # Forward fill or drop for other features
                        if missing_rate < 0.1:
                            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                        else:
                            print(f"WARNING: High missing rate for {col}, consider excluding")
        
        return df
    
    def create_train_test_splits(self, df: pd.DataFrame, strategy: str = "leave_one_city_out") -> List[Tuple]:
        """Create train/test splits based on strategy."""
        
        splits = []
        
        if strategy == "leave_one_city_out":
            cities = df['city'].unique()
            for test_city in cities:
                train_mask = df['city'] != test_city
                test_mask = df['city'] == test_city
                splits.append((train_mask, test_mask, test_city))
        
        elif strategy == "temporal_split":
            # Sort by datetime
            df_sorted = df.sort_values(['city', 'datetime'])
            
            for city in df['city'].unique():
                city_data = df_sorted[df_sorted['city'] == city]
                split_idx = int(len(city_data) * 0.8)
                
                train_indices = city_data.index[:split_idx]
                test_indices = city_data.index[split_idx:]
                
                train_mask = df.index.isin(train_indices)
                test_mask = df.index.isin(test_indices)
                splits.append((train_mask, test_mask, f"{city}_temporal"))
        
        elif strategy == "random_split":
            from sklearn.model_selection import train_test_split
            
            train_idx, test_idx = train_test_split(
                df.index, 
                test_size=0.2, 
                random_state=42, 
                stratify=df['city']
            )
            
            train_mask = df.index.isin(train_idx)
            test_mask = df.index.isin(test_idx)
            splits.append((train_mask, test_mask, "random"))
        
        return splits
    
    def train_model(self, 
                   X_train: pd.DataFrame, 
                   y_train: pd.Series,
                   feature_names: List[str],
                   task_type: str = "regression") -> Any:
        """Train CatBoost model."""
        
        model_config = self.config['training']['models']['catboost'].copy()
        
        # Identify categorical features
        categorical_features = []
        categorical_candidates = model_config.get('categorical_features', [])
        
        for cat_feature in categorical_candidates:
            if cat_feature in feature_names:
                categorical_features.append(cat_feature)
        
        print(f"Categorical features: {categorical_features}")
        
        # Initialize model
        if task_type == "regression":
            model = CatBoostRegressor(**{k: v for k, v in model_config.items() 
                                       if k != 'categorical_features'})
        else:
            model_config['loss_function'] = 'MultiClass'
            model_config['eval_metric'] = 'Accuracy'
            model = CatBoostClassifier(**{k: v for k, v in model_config.items() 
                                        if k != 'categorical_features'})
        
        # Create training pool
        train_pool = Pool(
            data=X_train[feature_names],
            label=y_train,
            cat_features=categorical_features
        )
        
        # Train model
        print("Training model...")
        start_time = time.time()
        
        model.fit(
            train_pool,
            verbose=100  # Print progress every 100 iterations
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds")
        
        return model
    
    def evaluate_model(self, 
                      model: Any,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      feature_names: List[str],
                      task_type: str = "regression") -> Dict:
        """Evaluate model performance."""
        
        # Make predictions
        y_pred = model.predict(X_test[feature_names])
        
        metrics = {}
        
        if task_type == "regression":
            metrics.update({
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
            })
        else:
            metrics.update({
                'accuracy': balanced_accuracy_score(y_test, y_pred),
                'cohens_kappa': cohen_kappa_score(y_test, y_pred)
            })
        
        return metrics, y_pred
    
    def run_experiment(self, 
                      cities: List[str], 
                      feature_groups: List[str],
                      cv_strategy: str = "leave_one_city_out") -> Dict:
        """Run complete training experiment."""
        
        print(f"Starting experiment: cities={cities}, features={feature_groups}")
        
        # Load data
        feature_tables = self.find_feature_tables(cities)
        if not feature_tables:
            raise ValueError(f"No feature tables found for cities: {cities}")
        
        df = self.load_and_merge_data(feature_tables)
        
        # Prepare features
        feature_df, feature_names = self.prepare_features(df, feature_groups)
        
        # Create splits
        splits = self.create_train_test_splits(feature_df, cv_strategy)
        
        # Run cross-validation
        cv_results = []
        feature_importance_all = []
        
        for i, (train_mask, test_mask, split_name) in enumerate(splits):
            print(f"\nSplit {i+1}/{len(splits)}: {split_name}")
            
            # Prepare data
            X_train = feature_df[train_mask]
            X_test = feature_df[test_mask]
            y_train = X_train['count']
            y_test = X_test['count']
            
            print(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
            
            # Train model
            model = self.train_model(X_train, y_train, feature_names)
            
            # Evaluate
            metrics, y_pred = self.evaluate_model(model, X_test, y_test, feature_names)
            
            # Feature importance
            feature_importance = model.get_feature_importance()
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance,
                'split': split_name
            })
            feature_importance_all.append(importance_df)
            
            # Save results
            split_results = {
                'split_name': split_name,
                'metrics': metrics,
                'n_train': len(X_train),
                'n_test': len(X_test)
            }
            cv_results.append(split_results)
            
            # Save model for this split
            model_path = f"data/models/{self.run_id}_{split_name}.cbm"
            model.save_model(model_path)
            print(f"Saved model: {model_path}")
        
        # Aggregate results
        experiment_results = {
            'experiment_config': {
                'cities': cities,
                'feature_groups': feature_groups,
                'cv_strategy': cv_strategy,
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat()
            },
            'cv_results': cv_results,
            'feature_names': feature_names,
            'summary_metrics': self.summarize_cv_results(cv_results)
        }
        
        # Save feature importance
        if feature_importance_all:
            all_importance = pd.concat(feature_importance_all, ignore_index=True)
            importance_summary = all_importance.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
            importance_summary = importance_summary.sort_values('mean', ascending=False)
            
            importance_path = f"reports/feature_importance/{self.run_id}_importance.csv"
            importance_summary.to_csv(importance_path, index=False)
            print(f"Saved feature importance: {importance_path}")
            
            experiment_results['feature_importance_path'] = importance_path
        
        return experiment_results
    
    def summarize_cv_results(self, cv_results: List[Dict]) -> Dict:
        """Summarize cross-validation results."""
        
        # Extract metrics across all splits
        all_metrics = {}
        for result in cv_results:
            for metric, value in result['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Compute summary statistics
        summary = {}
        for metric, values in all_metrics.items():
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)
        
        return summary
    
    def save_results(self, results: Dict):
        """Save experiment results."""
        
        # Save metrics JSON
        metrics_path = f"reports/metrics/{self.run_id}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics: {metrics_path}")
        
        # Save run manifest
        manifest_path = f"reports/runs/{self.run_id}_manifest.md"
        self.create_run_manifest(results, manifest_path)
        print(f"Saved run manifest: {manifest_path}")
        
        # Update results
        self.results = results
    
    def create_run_manifest(self, results: Dict, output_path: str):
        """Create human-readable run manifest."""
        
        with open(output_path, 'w') as f:
            f.write(f"# Training Run Manifest\n\n")
            f.write(f"**Run ID**: {results['experiment_config']['run_id']}\n")
            f.write(f"**Timestamp**: {results['experiment_config']['timestamp']}\n")
            f.write(f"**Cities**: {', '.join(results['experiment_config']['cities'])}\n")
            f.write(f"**Features**: {' + '.join(results['experiment_config']['feature_groups'])}\n")
            f.write(f"**CV Strategy**: {results['experiment_config']['cv_strategy']}\n\n")
            
            f.write("## Summary Metrics\n\n")
            for metric, value in results['summary_metrics'].items():
                f.write(f"- **{metric}**: {value:.4f}\n")
            
            f.write("\n## Cross-Validation Results\n\n")
            f.write("| Split | RMSE | MAE | R² | MAPE |\n")
            f.write("|-------|------|-----|----|----- |\n")
            
            for result in results['cv_results']:
                split_name = result['split_name']
                metrics = result['metrics']
                f.write(f"| {split_name} | {metrics.get('rmse', 0):.2f} | "
                       f"{metrics.get('mae', 0):.2f} | {metrics.get('r2', 0):.3f} | "
                       f"{metrics.get('mape', 0):.1f}% |\n")
            
            f.write(f"\n## Features Used\n\n")
            for i, feature in enumerate(results['feature_names'], 1):
                f.write(f"{i}. {feature}\n")


def main():
    parser = argparse.ArgumentParser(description="Train pedestrian volume prediction models")
    parser.add_argument("--cities", nargs='+', required=True,
                       help="Cities to include in training (melbourne, new_york, zurich, dublin)")
    parser.add_argument("--features", default="baseline+environmental",
                       help="Feature groups to use (e.g., baseline+environmental+network)")
    parser.add_argument("--cv-strategy", default="leave_one_city_out",
                       choices=["leave_one_city_out", "temporal_split", "random_split"],
                       help="Cross-validation strategy")
    parser.add_argument("--config", default="config/cities.yaml",
                       help="Configuration file path")
    parser.add_argument("--year", type=int,
                       help="Override default year for all cities")
    
    args = parser.parse_args()
    
    # Parse feature groups
    feature_groups = args.features.split('+')
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(args.config)
    
    try:
        # Run experiment
        results = orchestrator.run_experiment(
            cities=args.cities,
            feature_groups=feature_groups,
            cv_strategy=args.cv_strategy
        )
        
        # Save results
        orchestrator.save_results(results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Run ID: {results['experiment_config']['run_id']}")
        print(f"Cities: {', '.join(args.cities)}")
        print(f"Features: {args.features}")
        print(f"CV Strategy: {args.cv_strategy}")
        print()
        
        summary = results['summary_metrics']
        if 'rmse_mean' in summary:
            print(f"Average RMSE: {summary['rmse_mean']:.2f} ± {summary['rmse_std']:.2f}")
            print(f"Average MAE:  {summary['mae_mean']:.2f} ± {summary['mae_std']:.2f}")
            print(f"Average R²:   {summary['r2_mean']:.3f} ± {summary['r2_std']:.3f}")
            print(f"Average MAPE: {summary['mape_mean']:.1f}% ± {summary['mape_std']:.1f}%")
        
        print(f"\nResults saved:")
        print(f"  Metrics: reports/metrics/{results['experiment_config']['run_id']}_metrics.json")
        print(f"  Manifest: reports/runs/{results['experiment_config']['run_id']}_manifest.md")
        if 'feature_importance_path' in results:
            print(f"  Feature Importance: {results['feature_importance_path']}")
        
        # Performance validation
        if summary.get('r2_mean', 0) < 0.3:
            print("\nWARNING: Average R² < 0.3 - consider feature engineering or data quality checks")
        
        if summary.get('rmse_std', 0) > summary.get('rmse_mean', 0) * 0.5:
            print("WARNING: High RMSE variance - model may not generalize well")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()