from flask import Flask, request, jsonify
from flask_cors import CORS
import osmnx as ox
import geopandas as gpd
import networkx as nx
import pandas as pd
from catboost import CatBoostClassifier

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load the CatBoost model (make sure cb_model.cbm is in the same folder)
model = CatBoostClassifier()
model.load_model("cb_model.cbm")

# Define which features and categorical columns the model expects
FEATS = [
    "length",        # edge length (meters)
    "betweenness",   # betweenness centrality
    "closeness",     # closeness centrality
    "Hour",          # hour of day
    "is_weekend",    # boolean weekend flag
    "time_of_day",   # categorical: morning/afternoon/evening/night
    "land_use",      # categorical land use class
    "highway"        # categorical highway type
]
CAT_COLS = ["time_of_day", "land_use", "highway"]


def extract_features(G, dt_index):
    """
    Given an OSMnx graph G and a datetime pandas.Index dt_index for features,
    compute a GeoDataFrame of edges with all required features.

    dt_index should contain a single pandas.Timestamp representing now, or
    a series of datetimes to apply temporal features.
    """
    # Convert to GeoDataFrame of edges
    gdf = ox.graph_to_gdfs(G, nodes=False)

    # 1) Length is already present
    gdf["length"] = gdf["length"].fillna(0)

    # 2) Centrality measures (approximate if G is large)
    # Using networkx approximation with k = min(500, n_nodes)
    n = len(G.nodes)
    k = min(500, n)
    between = nx.betweenness_centrality(G, k=k, normalized=True)
    close = nx.closeness_centrality(G)
    # Map centralities to edges via u->v
    gdf["betweenness"] = gdf["u"].map(between)
    gdf["closeness"]   = gdf["u"].map(close)

    # 3) Temporal features (using dt_index[0])
    ts = pd.to_datetime(dt_index[0])
    gdf["Hour"] = ts.hour
    gdf["is_weekend"] = int(ts.weekday() >= 5)
    # Derive time_of_day category
    h = ts.hour
    if 5 <= h < 12:
        tod = "morning"
    elif 12 <= h < 17:
        tod = "afternoon"
    elif 17 <= h < 21:
        tod = "evening"
    else:
        tod = "night"
    gdf["time_of_day"] = tod

    # 4) Categorical OSM tags
    gdf["land_use"] = gdf["landuse"].fillna("other").astype(str)
    gdf["highway"]  = gdf["highway"].fillna("unknown").astype(str)

    # Keep only required columns
    feats = gdf[FEATS].copy()

    # Cast categorical columns
    for c in CAT_COLS:
        feats[c] = feats[c].astype("category")

    return gdf, feats


@app.route("/predict", methods=["GET"])
def predict():
    place = request.args.get("place")
    date  = request.args.get("date")  # optional ISO string

    if not place:
        return jsonify({"error": "Missing 'place' parameter"}), 400

    try:
        # Load walkable graph from OSM
        G = ox.graph_from_place(place, network_type="walk")

        # Build a datetime index
        if date:
            dt_idx = pd.to_datetime([date])
        else:
            dt_idx = pd.to_datetime([pd.Timestamp.now()])

        # Extract features and original geometry frame
        gdf_edges, feats = extract_features(G, dt_idx)

        # Predict volume bins
        preds = model.predict(feats)
        gdf_edges["volume_bin"] = preds.astype(int)

        # Return GeoJSON of edges with volume_bin
        return gdf_edges.to_json()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
