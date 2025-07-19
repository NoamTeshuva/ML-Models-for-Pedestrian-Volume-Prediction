#!/usr/bin/env python3
# pedestrian-api/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import osmnx as ox
import pandas as pd
from catboost import CatBoostClassifier, Pool
import traceback

# Import your feature‑engineering functions from the local folder
from feature_engineering.centrality_features import compute_centrality
from feature_engineering.time_features       import compute_time_features
from feature_engineering.landuse_features import compute_landuse_edges
from feature_engineering.highway_features    import compute_highway

app = Flask(__name__)
CORS(app)

# Load the pre‑trained CatBoost model (cb_model.cbm must be present)
model = CatBoostClassifier()
model.load_model("cb_model.cbm")

# The features your model expects, in order
FEATS = [
    "length",
    "betweenness",
    "closeness",
    "Hour",
    "is_weekend",
    "time_of_day",
    "land_use",
    "highway",
]
CAT_COLS = ["time_of_day", "land_use", "highway"]


def extract_features(G, dt_index):
    # 1) Base GeoDataFrame
    gdf = ox.graph_to_gdfs(G, nodes=False).reset_index()
    print("🔹 BASE edges:", len(gdf))

    # 2) Centrality
    gdf = compute_centrality(G, gdf)
    print("🔹 After centrality:", gdf.shape)

    # 3) Temporal
    ts = pd.to_datetime(dt_index[0])
    gdf = compute_time_features(gdf, ts)
    print("🔹 After time feats:", gdf.shape)

    # 4) Land‑use
    gdf = compute_landuse_edges(gdf)
    print("🔹 After landuse:", gdf.shape)

    # 5) Highway
    gdf = compute_highway(gdf)
    print("🔹 After highway:", gdf.shape)

    # 6) Final feature set
    feats = gdf[FEATS].copy()
    print("🔹 Final feature df:", feats.shape)
    return gdf, feats


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    return jsonify({"pong": True})


@app.route("/predict", methods=["GET"])
def predict():
    """Predict pedestrian-volume bins (1–5) for the given place and date."""
    print(f"🚀 Received request with args: {request.args}")
    place = request.args.get("place")
    date  = request.args.get("date")

    if not place:
        return jsonify({"error": "Missing 'place' parameter"}), 400

    try:
        # 1) Download walkable graph
        G = ox.graph_from_place(place, network_type="walk")

        # 2) Build datetime index
        if date:
            dt_idx = pd.to_datetime([date])
        else:
            dt_idx = pd.to_datetime([pd.Timestamp.now()])

        # 3) Extract features
        gdf_edges, feats = extract_features(G, dt_idx)

        # 4) Predict and attach
        # Validate that all required features are present
        missing_features = [feat for feat in FEATS if feat not in feats.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Get indices of categorical features for CatBoost
        cat_feature_indices = [feats.columns.get_loc(col) for col in CAT_COLS if col in feats.columns]
        print(f"🔹 Categorical features: {CAT_COLS}")
        print(f"🔹 Categorical indices: {cat_feature_indices}")
        print(f"🔹 Feature columns: {list(feats.columns)}")
        print(f"🔹 Feature dtypes: {feats.dtypes.to_dict()}")
        
        # Ensure categorical features are strings
        for col in CAT_COLS:
            if col in feats.columns:
                feats[col] = feats[col].astype(str)
        
        # Create CatBoost Pool with categorical features
        pool = Pool(feats, cat_features=cat_feature_indices)
        
        # Make prediction using the pool
        preds = model.predict(pool)
        gdf_edges["volume_bin"] = preds.astype(int)

        # 5) Return GeoJSON
        return gdf_edges.to_json()

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Development server; use Gunicorn in production
    app.run(host="0.0.0.0", port=5000, debug=True)
