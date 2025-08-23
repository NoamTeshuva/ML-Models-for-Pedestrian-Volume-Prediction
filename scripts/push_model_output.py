#!/usr/bin/env python3
"""
push_model_output.py - Push pedestrian model predictions to ArcGIS Feature Layer

This script bridges the gap between the pedestrian prediction API and ArcGIS Experience Builder.
It fetches predictions from the API and pushes them to a hosted feature layer for visualization.

Approach: Full refresh (truncate then add) - clears existing features and loads fresh predictions.

SETUP:
1. Get your ArcGIS token from https://developers.arcgis.com/rest/users-groups-and-items/authentication.htm
2. Set environment variable: export ARCGIS_TOKEN="your_token_here"
3. Update FEATURE_LAYER_URL below with your Model_Output layer URL
4. Run: python scripts/push_model_output.py

RATE LIMITS:
- ArcGIS: 6000 features per request (we batch at 1000 for safety)
- OSM/Overpass: May throttle if requesting too frequently (wait 30s between runs if needed)
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Any
import logging

# ============ CONFIGURATION ============
API_BASE = "http://localhost:5000"
PLACE = "Tel Aviv, Israel"  # Change this to any city in Israel
MAX_FEATURES = 1500  # Adjust based on city size (Tel Aviv can handle ~1500-2000)

# IMPORTANT: Replace with your actual Model_Output feature layer URL
# It should end with /0 or /1 (the layer index)
FEATURE_LAYER_URL = "https://services6.arcgis.com/zq4jVznZCyIRsEqh/arcgis/rest/services/Model_Output/FeatureServer/0"

# Batch size for ArcGIS operations (max 6000, but 1000 is safer)
BATCH_SIZE = 1000

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ HELPER FUNCTIONS ============

def get_token():
    """Get ArcGIS token from command line, environment variable or prompt user."""
    # First try command line argument
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    # Then try environment variable
    token = os.environ.get('ARCGIS_TOKEN')
    if not token:
        logger.warning("ARCGIS_TOKEN not found in environment variables")
        token = input("Enter your ArcGIS token: ").strip()
    return token


def transform_geojson_to_features(geojson_data: Dict) -> List[Dict]:
    """
    Transform GeoJSON from the API to ArcGIS Feature Service format.
    
    Expected fields from /simulate:
    - edge_id, osmid, highway, length
    - pred_before, pred_after, delta
    - geometry (LineString)
    """
    features = []
    
    for feature in geojson_data.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        
        # Skip if no geometry
        if not geom or geom.get('type') != 'LineString':
            continue
        
        # Transform geometry to ArcGIS format
        coords = geom.get('coordinates', [])
        if not coords:
            continue
            
        # ArcGIS polyline format
        arcgis_geom = {
            "paths": [coords],
            "spatialReference": {"wkid": 4326}  # WGS84
        }
        
        # Build attributes (handle missing fields gracefully)
        attributes = {
            "edge_id": str(props.get('edge_id', '')),
            "osmid": str(props.get('osmid', ''))[:50] if props.get('osmid') else None,  # Truncate if too long
            "highway": str(props.get('highway', 'unclassified'))[:50],
            "length": float(props.get('length', 0)),
            "pred_before": float(props.get('pred_before', 0)),
            "pred_after": float(props.get('pred_after', 0)),
            "delta": float(props.get('delta', 0))
        }
        
        features.append({
            "attributes": attributes,
            "geometry": arcgis_geom
        })
    
    return features


def truncate_layer(layer_url: str, token: str) -> bool:
    """Delete all existing features from the layer."""
    delete_url = f"{layer_url}/deleteFeatures"
    params = {
        'f': 'json',
        'token': token,
        'where': '1=1'  # Delete all
    }
    
    logger.info("Truncating existing features...")
    response = requests.post(delete_url, data=params)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success') or 'deleteResults' in result:
            logger.info("Successfully truncated layer")
            return True
        else:
            logger.error(f"Truncate failed: {result}")
            return False
    else:
        logger.error(f"HTTP error during truncate: {response.status_code}")
        return False


def add_features_batch(layer_url: str, features: List[Dict], token: str) -> bool:
    """Add a batch of features to the layer."""
    add_url = f"{layer_url}/addFeatures"
    
    params = {
        'f': 'json',
        'token': token,
        'features': json.dumps(features)
    }
    
    response = requests.post(add_url, data=params)
    
    if response.status_code == 200:
        result = response.json()
        if 'addResults' in result:
            # Check how many succeeded
            success_count = sum(1 for r in result['addResults'] if r.get('success'))
            total_count = len(result['addResults'])
            
            if success_count == total_count:
                logger.info(f"Successfully added {success_count} features")
                return True
            else:
                logger.warning(f"Partial success: {success_count}/{total_count} features added")
                # Log specific errors
                for i, r in enumerate(result['addResults']):
                    if not r.get('success'):
                        logger.error(f"Feature {i} failed: {r.get('error')}")
                return success_count > 0
        else:
            logger.error(f"Unexpected response: {result}")
            return False
    else:
        logger.error(f"HTTP error: {response.status_code}")
        return False


def push_to_arcgis(layer_url: str, features: List[Dict], token: str) -> bool:
    """Push features to ArcGIS in batches."""
    if not features:
        logger.warning("No features to push")
        return False
    
    logger.info(f"Pushing {len(features)} features to ArcGIS...")
    
    # Process in batches
    success = True
    for i in range(0, len(features), BATCH_SIZE):
        batch = features[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = ((len(features) - 1) // BATCH_SIZE) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} features)")
        
        if not add_features_batch(layer_url, batch, token):
            success = False
            logger.error(f"Failed to add batch {batch_num}")
            # Continue with other batches even if one fails
        
        # Small delay between batches to avoid rate limiting
        if i + BATCH_SIZE < len(features):
            time.sleep(0.5)
    
    return success


def run_simulation(place: str, max_features: int, edits: List[Dict] = None) -> Dict:
    """
    Call the /simulate endpoint to get predictions.
    
    Args:
        place: City name (e.g., "Tel Aviv, Israel")
        max_features: Maximum number of features to return
        edits: Optional list of edit operations
    
    Returns:
        GeoJSON response from the API
    """
    url = f"{API_BASE}/simulate"
    
    payload = {
        "place": place,
        "max_features": max_features,
        "edits": edits or []
    }
    
    logger.info(f"Calling API for {place} (max {max_features} features)...")
    
    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("API request timed out (180s)")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise


def main():
    """Main execution flow."""
    print("=" * 60)
    print("Pedestrian Model -> ArcGIS Pipeline")
    print("=" * 60)
    print(f"API: {API_BASE}")
    print(f"Place: {PLACE}")
    print(f"Max Features: {MAX_FEATURES}")
    print(f"Target Layer: {FEATURE_LAYER_URL}")
    print("=" * 60)
    
    # Check configuration
    if "YOUR_ORG_ID" in FEATURE_LAYER_URL:
        logger.error("ERROR: You must update FEATURE_LAYER_URL with your actual feature layer URL!")
        logger.error("Find it in ArcGIS Online: Content → Model_Output → View Service URL")
        sys.exit(1)
    
    # Get token
    token = get_token()
    if not token:
        logger.error("No token provided. Exiting.")
        sys.exit(1)
    
    try:
        # Step 1: Call the simulation API
        logger.info("Step 1: Fetching predictions from API...")
        start_time = time.time()
        
        # For now, we're not applying any edits - just getting base predictions
        # In the full app, edits would come from the Student_Edges layer
        geojson_data = run_simulation(PLACE, MAX_FEATURES)
        
        api_time = time.time() - start_time
        feature_count = len(geojson_data.get('features', []))
        logger.info(f"Received {feature_count} features in {api_time:.2f}s")
        
        if feature_count == 0:
            logger.warning("No features returned from API. Check place name or try increasing max_features.")
            sys.exit(1)
        
        # Step 2: Transform to ArcGIS format
        logger.info("Step 2: Transforming to ArcGIS format...")
        features = transform_geojson_to_features(geojson_data)
        logger.info(f"Transformed {len(features)} features")
        
        # Step 3: Truncate existing features
        logger.info("Step 3: Clearing existing features...")
        if not truncate_layer(FEATURE_LAYER_URL, token):
            logger.warning("Truncate may have failed, continuing anyway...")
        
        # Step 4: Push new features
        logger.info("Step 4: Pushing new features...")
        push_start = time.time()
        
        if push_to_arcgis(FEATURE_LAYER_URL, features, token):
            push_time = time.time() - push_start
            total_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"Features pushed: {len(features)}")
            print(f"API call time: {api_time:.2f}s")
            print(f"Push time: {push_time:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            print("\nNext steps:")
            print("1. Open your Experience app")
            print("2. Toggle 'Model_Output view' in the Layer List")
            print("3. You should see colored lines (styled by delta)")
            print("4. Click on lines to see pop-ups with pred_before, pred_after, delta")
        else:
            logger.error("Failed to push all features to ArcGIS")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# ============ README ============
"""
QUICK START:
-----------
1. Update FEATURE_LAYER_URL above with your actual URL
2. Set your token: export ARCGIS_TOKEN="your_token_here"
3. Run: python scripts/push_model_output.py

CHANGING THE PLACE:
------------------
Edit the PLACE variable at the top of this script.
Examples for Israel:
- "Tel Aviv, Israel"
- "Jerusalem, Israel"
- "Haifa, Israel"
- "Beer Sheva, Israel"
- "Petah Tikva, Israel"

TROUBLESHOOTING:
---------------
If you get rate limited:
- By ArcGIS: Reduce BATCH_SIZE to 500
- By OSM/Overpass: Wait 30 seconds between runs

If features don't appear:
1. Check the Experience map refresh button
2. Verify Model_Output view layer is visible
3. Check that symbology is set on the delta field
4. Try zooming to the city extent

INTEGRATING WITH STUDENT EDITS:
-------------------------------
To incorporate actual student edits from Student_Edges layer:
1. Query Student_Edges layer for current edits
2. Transform to edit operations format
3. Pass edits array to run_simulation() function
4. The API will compute pred_before (original) vs pred_after (with edits)

Example edit format:
edits = [
    {"op": "delete", "edge_id": "e_001234"},
    {"op": "add", "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]}, "highway": "pedestrian"},
    {"op": "reshape", "edge_id": "e_005678", "geometry": {"type": "LineString", "coordinates": [...]}}
]
"""