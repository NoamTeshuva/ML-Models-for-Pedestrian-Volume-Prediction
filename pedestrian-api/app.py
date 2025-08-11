#!/usr/bin/env python3
# pedestrian-api/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Optional, Tuple
from catboost import CatBoostClassifier, Pool

# Import the unified feature pipeline
from feature_engineering.feature_pipeline import (
    run_feature_pipeline,
    prepare_model_features,
    PipelineError,
    PipelineConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

app = Flask(__name__)
CORS(app)

# Load the pre-trained CatBoost model
try:
    model = CatBoostClassifier()
    model.load_model("cb_model.cbm")
    logging.info("Successfully loaded CatBoost model")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Use configuration from pipeline
FEATS = PipelineConfig.FEATURE_COLUMNS
CAT_COLS = PipelineConfig.CATEGORICAL_COLUMNS


def validate_request_params(place: Optional[str], bbox_str: Optional[str], date: Optional[str]) -> Tuple[Optional[str], Optional[Tuple[float, float, float, float]], Optional[str]]:
    """Validate and parse request parameters.
    
    Args:
        place: Place name parameter
        bbox_str: Bounding box string parameter
        date: Date string parameter
        
    Returns:
        tuple: (validated_place, parsed_bbox, validated_date)
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate place or bbox
    if not place and not bbox_str:
        raise ValueError("Either 'place' or 'bbox' parameter is required")
    
    # Parse and validate bbox
    bbox = None
    if bbox_str:
        try:
            bbox_parts = [float(x.strip()) for x in bbox_str.split(",")]
            if len(bbox_parts) != 4:
                raise ValueError("Bbox must contain exactly 4 coordinates")
            bbox = tuple(bbox_parts)
            
            # Validate coordinate ranges
            minx, miny, maxx, maxy = bbox
            if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
                raise ValueError("Longitude values must be between -180 and 180")
            if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
                raise ValueError("Latitude values must be between -90 and 90")
            if minx >= maxx or miny >= maxy:
                raise ValueError("Invalid bbox: min values must be less than max values")
                
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid bbox format: {e}")
    
    # Validate place
    if place and not isinstance(place, str):
        raise ValueError("Place parameter must be a string")
    
    # Validate date
    if date:
        try:
            pd.to_datetime(date)
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
    
    return place, bbox, date


def create_prediction_response(features_gdf, predictions, metadata) -> dict:
    """Create structured prediction response.
    
    Args:
        features_gdf: GeoDataFrame with features and predictions
        predictions: Model predictions array
        metadata: Pipeline processing metadata
        
    Returns:
        dict: Structured JSON response
    """
    # Add predictions to features
    features_gdf = features_gdf.copy()
    features_gdf["volume_bin"] = predictions.astype(int)
    
    # Get sample features for response (convert numpy types to native Python types)
    sample_features = {}
    if len(features_gdf) > 0:
        sample_row = features_gdf.iloc[0]
        for col in FEATS:
            if col in sample_row:
                val = sample_row[col]
                # Convert numpy/pandas types to JSON-serializable types
                if hasattr(val, 'item'):  # numpy scalars
                    sample_features[col] = val.item()
                elif pd.api.types.is_integer_dtype(type(val)):
                    sample_features[col] = int(val)
                elif pd.api.types.is_float_dtype(type(val)):
                    sample_features[col] = float(val)
                else:
                    sample_features[col] = str(val)
    
    # Convert metadata values to JSON-serializable types
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif pd.api.types.is_integer_dtype(type(obj)):
            return int(obj)
        elif pd.api.types.is_float_dtype(type(obj)):
            return float(obj)
        else:
            return obj
    
    # Create response
    response = {
        "success": True,
        "location": metadata["location"],
        "timestamp": metadata["timestamp"],
        "processing_time": round(float(metadata["processing_time"]), 2),
        "network_stats": make_json_serializable(metadata["network_stats"]),
        "sample_prediction": {
            "volume_bin": int(predictions[0]) if len(predictions) > 0 else None,
            "features": sample_features
        },
        "validation": make_json_serializable(metadata["validation"]),
        "geojson": features_gdf.__geo_interface__
    }
    
    return response


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    return jsonify({"pong": True})


@app.route("/predict", methods=["GET"])
def predict():
    """Predict pedestrian volume bins (1-5) for the given place and date.
    
    Query Parameters:
        place (str, optional): Place name (e.g., "Monaco", "Tel Aviv")
        bbox (str, optional): Bounding box as "minx,miny,maxx,maxy"
        date (str, optional): ISO timestamp (defaults to current time)
        
    Returns:
        JSON response with predictions and metadata
    """
    request_start = time.time()
    
    # Get request parameters
    place = request.args.get("place")
    date = request.args.get("date")
    bbox_str = request.args.get("bbox")
    
    logging.info(f"Received prediction request: place={place}, bbox={bbox_str}, date={date}")
    
    try:
        # 1. Validate request parameters
        place, bbox, date = validate_request_params(place, bbox_str, date)
        
        # 2. Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not available",
                "code": 503,
                "details": "CatBoost model failed to load at startup"
            }), 503
        
        # 3. Run feature extraction pipeline
        features_gdf, pipeline_metadata = run_feature_pipeline(
            place=place,
            bbox=bbox,
            timestamp=date
        )
        
        # 4. Prepare features for model
        model_features = prepare_model_features(features_gdf)
        
        # 5. Make predictions
        logging.info(f"Making predictions for {len(model_features)} edges")
        
        # Get categorical feature indices
        cat_feature_indices = [model_features.columns.get_loc(col) for col in CAT_COLS if col in model_features.columns]
        
        # Ensure categorical features are strings
        for col in CAT_COLS:
            if col in model_features.columns:
                model_features[col] = model_features[col].astype(str)
        
        # Create CatBoost Pool and predict
        pool = Pool(model_features, cat_features=cat_feature_indices)
        predictions = model.predict(pool)
        
        # 6. Create response
        total_time = time.time() - request_start
        pipeline_metadata["total_request_time"] = total_time
        
        response = create_prediction_response(features_gdf, predictions, pipeline_metadata)
        
        logging.info(f"Prediction completed in {total_time:.2f}s for {len(predictions)} edges")
        
        return jsonify(response)
        
    except PipelineError as e:
        logging.error(f"Pipeline error: {e.message}")
        return jsonify(e.to_dict()), e.code
        
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "code": 400,
            "details": {"place": place, "bbox": bbox_str, "date": date}
        }), 400
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "code": 500,
            "details": str(e)
        }), 500


if __name__ == "__main__":
    # Development server; use Gunicorn in production
    import os
    debug_mode = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(host="0.0.0.0", port=int(os.getenv('PORT', 5000)), debug=debug_mode)
