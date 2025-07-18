**Pedestrian Volume Prediction Project**

---

## Overview

This repository provides tools and services to predict pedestrian-volume levels (1–5) on street segments, leveraging machine learning models trained on Melbourne 2023 data and tested on New York City 2023 data. It includes:

* **Data processing pipelines** to load, bin, and engineer features from raw count and OpenStreetMap (OSM) data.
* **Model training scripts** using CatBoost with equal-frequency volume binning.
* **A Flask-based API** for on‑demand volume prediction by place name.

The goal is to enable real-time pedestrian-volume estimation in any place supported by OSM, even where sensor data is lacking.

---

## Project Structure

```text
pedestrian-volume-prediction/
├── data/                                  # Raw and processed data
│   ├── processed/
│   │   ├── melbourne/csv/                  # 2023 feature table for Melbourne
│   │   └── NewYork/csv/                    # 2023 feature table for NYC
│   └── raw/                               # (Optional) raw CSVs or downloads
├── src/                                   # Core code
│   ├── feature_engineering/               # Scripts to extract features:
│   │   ├── centrality_features.py         # Betweenness & closeness
│   │   ├── time_features.py               # Hour, weekend flag, time_of_day
│   │   ├── land_use_features.py           # land_use classification
│   │   └── highway_features.py            # highway type extraction
│   ├── models/                            # Model training and saved files
│   │   ├── train_catboost_volume_classifier.py  # Train & save CatBoost model
│   │   ├── cb_model.cbm                   # Trained CatBoost model file
│   └── api/                               # Flask service deployment
│       ├── app.py                         # Flask API for `/predict`
│       ├── requirements.txt               # Python dependencies
│       └── Procfile                       # Startup command for Render/Heroku
└── README.md                              # Project documentation
```

---

## Data Engineering

1. **Load and Binning**:

   * Use `pandas.qcut` for equal-frequency binning of raw pedestrian counts into five bins (1–5).
   * Fallback to equal-width bins if not enough unique values.

2. **Feature Extraction**:

   * **Centrality**: Approximate betweenness and closeness via NetworkX on the OSMnx graph.
   * **Temporal**: Extract hour of day, weekend indicator, and assign a time-of-day category (morning, afternoon, evening, night).
   * **OSM Tags**: Read `highway` and `landuse` attributes from edges to categorize street type and land use.
   * **Geometry**: Keep edge geometries for GeoJSON output.

Scripts are modular under `src/feature_engineering/` to allow reuse in both training and API.

---

## Model Training

The `train_catboost_volume_classifier.py` script:

* Loads processed feature tables for Melbourne (train) and NYC (test).
* Bins volume into equal-frequency categories.
* Defines features: `betweenness, closeness, Hour, is_weekend, time_of_day, land_use, highway`.
* Trains a `CatBoostClassifier` (800 iterations, depth=6, learning rate=0.05).
* Evaluates accuracy, Cohen’s κ, confusion matrix, classification report, MAE/MSE/RMSE/MedAE/R²/EVS on NYC.
* Saves the trained model as `cb_model.cbm`.

```bash
python src/models/train_catboost_volume_classifier.py
# Output: cb_model.cbm saved to src/models/
```

---

## API Service

A Flask-based microservice under `src/api/app.py` exposes a GET endpoint:

* **Endpoint**: `/predict`
* **Parameters**:

  * `place` (required): Place name to query OSMnx (e.g., `Tel Aviv, Israel`).
  * `date` (optional): ISO datetime string for temporal features; defaults to current timestamp.

**Example**:

```bash
curl "http://localhost:5000/predict?place=Tel+Aviv&date=2025-07-18T09:00:00"
```

**Response**: GeoJSON `FeatureCollection` of street edges, each with predicted `volume_bin` and original feature properties.

Dependencies are listed in `src/api/requirements.txt`:

```text
flask
flask-cors
catboost
pandas
numpy
geopandas
osmnx
shapely
networkx
```

For deployment, a `Procfile` runs:

```text
web: gunicorn app:app
```

---

## Local Testing

1. **Install dependencies**:

   ```bash
   cd src/api
   pip install -r requirements.txt
   ```
2. **Start the server**:

   ```bash
   python app.py
   ```
3. **Test endpoint**:
   Navigate to `http://127.0.0.1:5000/predict?place=Melbourne,Australia`.

---

## Deployment

1. **Push** the `src/api` folder (with `app.py`, `cb_model.cbm`, `requirements.txt`, `Procfile`) to GitHub.
2. **Create** a new Web Service on Render or Heroku.
3. **Link** the GitHub repo; the platform will detect dependencies and launch your API.
4. **Embed** the API in an ArcGIS Online Web AppBuilder or Experience Builder app using the Embed widget or custom JS (`GeoJSONLayer`).

---

## Contact

**Noam Teshuva**
CS Candidate, Lab Assistant
Civil Engineering Department, Ariel University
Email: [teshuva91@gmail.com](mailto:teshuva91@gmail.com)

**Supervisor**: Dr. Achituv Cohen

---

*Last updated: July 2025*
