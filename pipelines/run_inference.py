#!/usr/bin/env python3
"""
Inference Pipeline Orchestrator

Applies trained models to predict pedestrian volumes for new cities, sensors, or time periods.
Supports both sensor-level predictions and network-wide inference.

Usage:
    # Predict for existing sensors
    python pipelines/run_inference.py --city melbourne --model data/models/20231201_143530_melbourne.cbm --mode sensor --start-date 2023-12-01 --end-date 2023-12-07
    
    # Network-wide predictions
    python pipelines/run_inference.py --city new_city --model data/models/20231201_143530_cross_city.cbm --mode network --start-date 2023-12-01 --end-date 2023-12-01
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class InferenceOrchestrator:
    def __init__(self, config_path: str = "config/cities.yaml"):
        """Initialize inference orchestrator with configuration."""
        self.config = self.load_config(config_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        os.makedirs("reports/predictions", exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, model_path: str) -> Tuple[Any, Dict]:
        """Load trained model and its metadata."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = CatBoostRegressor()
        model.load_model(model_path)
        
        # Load metadata if available
        metadata_path = f"{model_path}.metadata.json"
        metadata = {}
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            print(f"Warning: No metadata found for model {model_path}")
            # Try to infer feature names from model
            try:
                metadata['feature_names'] = model.feature_names_
            except:
                metadata['feature_names'] = []
        
        return model, metadata
    
    def extract_features_for_city(self, 
                                 city: str, 
                                 start_date: str, 
                                 end_date: str,
                                 mode: str = "sensor") -> pd.DataFrame:
        """Extract features for a city and date range."""
        
        print(f"Extracting features for {city} ({start_date} to {end_date})")
        
        # Try to find existing feature table
        feature_table_path = self.find_city_feature_table(city)
        
        if feature_table_path and mode == "sensor":
            # Use existing sensor-level features
            features_df = self.load_sensor_features(feature_table_path, start_date, end_date)
        else:
            # Generate features from scratch
            features_df = self.generate_features_from_scratch(city, start_date, end_date, mode)
        
        return features_df
    
    def find_city_feature_table(self, city: str) -> Optional[str]:
        """Find existing feature table for a city."""
        city_config = self.config['cities'].get(city, {})
        target_year = city_config.get('default_year', self.config['defaults']['year'])
        
        possible_paths = [
            f"data/processed/{city}/csv/feature_table_{target_year}_{city}_with_env_topo_canopy.csv",
            f"data/processed/{city.title()}/csv/feature_table_{target_year}_{city}_with_env_topo_canopy.csv",
            f"data/processed/{city}/csv/feature_table_{target_year}_{city}_aligned.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found existing feature table: {path}")
                return path
        
        print(f"No existing feature table found for {city}")
        return None
    
    def load_sensor_features(self, 
                           feature_table_path: str, 
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """Load sensor features from existing feature table."""
        
        df = pd.read_csv(feature_table_path)
        
        # Filter by date range if dates are in the data
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        else:
            # Generate temporal grid
            df = self.expand_temporal_grid(df, start_date, end_date)
        
        return df
    
    def expand_temporal_grid(self, 
                           sensor_df: pd.DataFrame, 
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """Expand sensor locations to temporal grid."""
        
        # Create temporal grid
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        hour_range = range(24)
        
        expanded_data = []
        
        # Get unique sensors
        sensor_cols = ['sensor_id', 'sensor_name', 'latitude', 'longitude']
        available_sensor_cols = [col for col in sensor_cols if col in sensor_df.columns]
        
        if not available_sensor_cols:
            raise ValueError("No sensor identification columns found")
        
        unique_sensors = sensor_df[available_sensor_cols].drop_duplicates()
        
        for _, sensor in unique_sensors.iterrows():
            for date in date_range:
                for hour in hour_range:
                    row = sensor.to_dict()
                    row.update({
                        'date': date.strftime('%Y-%m-%d'),
                        'time': f"{hour:02d}:00",
                        'datetime': date + timedelta(hours=hour),
                        'hour': hour,
                        'day_of_week': date.dayofweek,
                        'month': date.month,
                        'is_weekend': date.dayofweek >= 5
                    })
                    
                    # Add environmental features if available
                    env_features = ['mean_grade_pct', 'max_grade_pct', 'elev_start_m', 
                                  'elev_end_m', 'elev_range_m', 'canopy_pct_buffer']
                    
                    for env_feat in env_features:
                        if env_feat in sensor_df.columns:
                            # Use the sensor's environmental feature value
                            sensor_mask = True
                            for col in available_sensor_cols:
                                if col in sensor and col in sensor_df.columns:
                                    sensor_mask &= sensor_df[col] == sensor[col]
                            
                            if sensor_mask.any():
                                row[env_feat] = sensor_df.loc[sensor_mask, env_feat].iloc[0]
                    
                    expanded_data.append(row)
        
        return pd.DataFrame(expanded_data)
    
    def generate_features_from_scratch(self, 
                                     city: str, 
                                     start_date: str, 
                                     end_date: str,
                                     mode: str) -> pd.DataFrame:
        """Generate features from scratch for new city."""
        
        print(f"Generating features from scratch for {city}")
        
        if mode == "network":
            # Network-wide inference: need to extract features for all edges
            return self.extract_network_features(city, start_date, end_date)
        else:
            # Sensor mode: need sensor locations
            sensor_path = self.find_sensor_file(city)
            if not sensor_path:
                raise ValueError(f"No sensor file found for {city} and no existing features available")
            
            return self.extract_sensor_features(city, sensor_path, start_date, end_date)
    
    def find_sensor_file(self, city: str) -> Optional[str]:
        """Find sensor file for a city."""
        city_config = self.config['cities'].get(city, {})
        sensor_path = city_config.get('paths', {}).get('sensors_csv')
        
        if sensor_path and os.path.exists(sensor_path):
            return sensor_path
        
        # Try common patterns
        possible_paths = [
            f"data/raw/{city}/sensors.csv",
            f"data/raw/{city}/sensor_locations.csv",
            f"data/processed/{city}/csv/sensor_locations.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def extract_sensor_features(self, 
                              city: str, 
                              sensor_path: str, 
                              start_date: str, 
                              end_date: str) -> pd.DataFrame:
        """Extract features for sensors using the existing pipeline."""
        
        print(f"Running feature extraction pipeline for {city}")
        
        # This would ideally call the existing feature extraction pipeline
        # For now, create a simplified version
        
        sensors_df = pd.read_csv(sensor_path)
        
        # Standardize sensor data
        city_config = self.config['cities'].get(city, {})
        sensor_schema = city_config.get('sensor_schema', {})
        
        column_mapping = {
            sensor_schema.get('id_column', 'sensor_id'): 'sensor_id',
            sensor_schema.get('name_column', 'sensor_name'): 'sensor_name',
            sensor_schema.get('lat_column', 'latitude'): 'latitude',
            sensor_schema.get('lon_column', 'longitude'): 'longitude'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in sensors_df.columns and old_col != new_col:
                sensors_df = sensors_df.rename(columns={old_col: new_col})
        
        # Generate temporal features
        expanded_df = self.expand_temporal_grid(sensors_df, start_date, end_date)
        
        # Add placeholder environmental features (would be extracted from rasters)
        env_features = ['mean_grade_pct', 'max_grade_pct', 'elev_start_m', 
                       'elev_end_m', 'elev_range_m', 'canopy_pct_buffer']
        
        for feat in env_features:
            if feat not in expanded_df.columns:
                # Use median values from config or set to NaN
                expanded_df[feat] = np.nan
        
        return expanded_df
    
    def extract_network_features(self, 
                               city: str, 
                               start_date: str, 
                               end_date: str) -> pd.DataFrame:
        """Extract features for entire street network."""
        
        print(f"Network-wide feature extraction for {city} not implemented in this stub")
        print("This would require:")
        print("1. Loading street network GPKG")
        print("2. Running environmental feature extraction")
        print("3. Creating edge-level temporal grid")
        print("4. Generating predictions for all edges")
        
        # Return empty DataFrame for now
        return pd.DataFrame()
    
    def prepare_inference_features(self, 
                                 features_df: pd.DataFrame, 
                                 model_metadata: Dict) -> pd.DataFrame:
        """Prepare features for model inference."""
        
        required_features = model_metadata.get('feature_names', [])
        
        if not required_features:
            print("Warning: No feature names in model metadata")
            return features_df
        
        # Check for missing features
        available_features = features_df.columns.tolist()
        missing_features = [f for f in required_features if f not in available_features]
        
        if missing_features:
            print(f"Missing features: {missing_features}")
            
            # Apply fallback strategies
            features_df = self.apply_feature_fallbacks(features_df, missing_features)
        
        # Ensure column order matches training
        try:
            features_df = features_df[required_features]
        except KeyError as e:
            print(f"Error selecting required features: {e}")
            # Use available features that match
            available_required = [f for f in required_features if f in features_df.columns]
            features_df = features_df[available_required]
            print(f"Using {len(available_required)}/{len(required_features)} required features")
        
        return features_df
    
    def apply_feature_fallbacks(self, 
                              features_df: pd.DataFrame, 
                              missing_features: List[str]) -> pd.DataFrame:
        """Apply fallback strategies for missing features."""
        
        fallback_strategy = self.config['inference']['fallbacks'].get('missing_env_features', 'median_impute')
        
        env_features = ['mean_grade_pct', 'max_grade_pct', 'elev_start_m', 
                       'elev_end_m', 'elev_range_m', 'canopy_pct_buffer']
        
        for feature in missing_features:
            if feature in env_features:
                if fallback_strategy == 'median_impute':
                    # Use global median values (would be loaded from training data)
                    median_values = {
                        'mean_grade_pct': 2.5,
                        'max_grade_pct': 8.0,
                        'elev_start_m': 50.0,
                        'elev_end_m': 50.0,
                        'elev_range_m': 5.0,
                        'canopy_pct_buffer': 0.3
                    }
                    features_df[feature] = median_values.get(feature, 0.0)
                    print(f"Imputed {feature} with median value: {median_values.get(feature, 0.0)}")
                
                else:
                    # Set to zero
                    features_df[feature] = 0.0
                    print(f"Set {feature} to 0.0")
            
            else:
                # For non-environmental features, set appropriate defaults
                if feature in ['highway']:
                    features_df[feature] = 'residential'  # Default road type
                elif feature in ['hour', 'day_of_week', 'month']:
                    pass  # These should be generated from date/time
                else:
                    features_df[feature] = 0.0
        
        return features_df
    
    def make_predictions(self, 
                        model: Any, 
                        features_df: pd.DataFrame,
                        include_confidence: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the trained model."""
        
        print(f"Making predictions for {len(features_df):,} samples")
        
        # Make predictions
        predictions = model.predict(features_df)
        
        # Estimate confidence/uncertainty
        confidence = None
        if include_confidence:
            try:
                # For regression, use prediction variance if available
                # This is a simplified confidence estimation
                prediction_std = np.std(predictions)
                confidence = 1.0 / (1.0 + prediction_std / np.mean(predictions))
                confidence = np.full(len(predictions), confidence)
                
                print(f"Estimated prediction confidence: {np.mean(confidence):.3f}")
            
            except Exception as e:
                print(f"Could not estimate confidence: {e}")
        
        return predictions, confidence
    
    def format_predictions(self, 
                          features_df: pd.DataFrame,
                          predictions: np.ndarray,
                          confidence: Optional[np.ndarray],
                          output_format: str = "csv") -> str:
        """Format predictions for output."""
        
        # Create results DataFrame
        results_df = features_df[['sensor_id', 'date', 'time']].copy()
        results_df['predicted_count'] = predictions
        
        if confidence is not None:
            results_df['confidence'] = confidence
        
        # Add metadata
        results_df['prediction_timestamp'] = datetime.now().isoformat()
        results_df['run_id'] = self.run_id
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "csv":
            output_path = f"reports/predictions/{timestamp}_{self.run_id}_predictions.csv"
            results_df.to_csv(output_path, index=False)
        
        elif output_format == "json":
            output_path = f"reports/predictions/{timestamp}_{self.run_id}_predictions.json"
            results_df.to_json(output_path, orient='records', date_format='iso')
        
        elif output_format == "geojson":
            # Convert to GeoDataFrame if coordinates available
            if 'latitude' in features_df.columns and 'longitude' in features_df.columns:
                from shapely.geometry import Point
                
                geometry = [Point(lon, lat) for lon, lat in 
                           zip(features_df['longitude'], features_df['latitude'])]
                
                geo_df = gpd.GeoDataFrame(results_df, geometry=geometry, crs='EPSG:4326')
                output_path = f"reports/predictions/{timestamp}_{self.run_id}_predictions.geojson"
                geo_df.to_file(output_path, driver='GeoJSON')
            else:
                print("Warning: No coordinates available for GeoJSON output, falling back to JSON")
                output_path = f"reports/predictions/{timestamp}_{self.run_id}_predictions.json"
                results_df.to_json(output_path, orient='records', date_format='iso')
        
        print(f"Saved predictions: {output_path}")
        return output_path
    
    def run_inference(self, 
                     city: str,
                     model_path: str,
                     start_date: str,
                     end_date: str,
                     mode: str = "sensor",
                     output_format: str = "csv") -> str:
        """Run complete inference pipeline."""
        
        print(f"Starting inference for {city}")
        print(f"Model: {model_path}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Mode: {mode}")
        
        # Load model
        model, model_metadata = self.load_model(model_path)
        
        # Extract features
        features_df = self.extract_features_for_city(city, start_date, end_date, mode)
        
        if features_df.empty:
            raise ValueError(f"No features extracted for {city}")
        
        print(f"Extracted features: {features_df.shape[0]:,} samples, {features_df.shape[1]} features")
        
        # Prepare features for inference
        inference_features = self.prepare_inference_features(features_df, model_metadata)
        
        # Make predictions
        predictions, confidence = self.make_predictions(model, inference_features)
        
        # Format and save results
        output_path = self.format_predictions(
            features_df, predictions, confidence, output_format
        )
        
        # Generate summary
        print(f"\n{'='*60}")
        print("INFERENCE SUMMARY")
        print(f"{'='*60}")
        print(f"City: {city}")
        print(f"Predictions: {len(predictions):,}")
        print(f"Mean predicted volume: {np.mean(predictions):.1f}")
        print(f"Min predicted volume: {np.min(predictions):.1f}")
        print(f"Max predicted volume: {np.max(predictions):.1f}")
        if confidence is not None:
            print(f"Average confidence: {np.mean(confidence):.3f}")
        print(f"Output: {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Run pedestrian volume inference")
    parser.add_argument("--city", required=True,
                       help="City to run inference for")
    parser.add_argument("--model", required=True,
                       help="Path to trained model file")
    parser.add_argument("--start-date", required=True,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--mode", default="sensor", choices=["sensor", "network"],
                       help="Inference mode: sensor or network-wide")
    parser.add_argument("--output-format", default="csv", 
                       choices=["csv", "json", "geojson"],
                       help="Output format")
    parser.add_argument("--config", default="config/cities.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = InferenceOrchestrator(args.config)
    
    try:
        # Run inference
        output_path = orchestrator.run_inference(
            city=args.city,
            model_path=args.model,
            start_date=args.start_date,
            end_date=args.end_date,
            mode=args.mode,
            output_format=args.output_format
        )
        
        print(f"\nInference completed successfully!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()