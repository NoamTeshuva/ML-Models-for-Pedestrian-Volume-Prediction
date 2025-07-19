# Pedestrian Volume Prediction API - Project Scope

## Project Overview

We're building a Flask-based REST API that predicts pedestrian volume bins (1-5) for street segments in any city. Given a place name and optional timestamp, the API returns a GeoJSON of street segments annotated with predicted pedestrian volume levels.

### Core Workflow
1. **Pull OSM walkable graph** for the city via osmnx
2. **Extract spatial features** (length, centrality, land-use, highway type)
3. **Extract temporal features** (hour, weekend flag, time-of-day)
4. **Feed features into pre-trained CatBoost model**
5. **Return GeoJSON** of edges augmented with "volume_bin" predictions

## Code Architecture

### Main Application (`pedestrian-api/app.py`)
- **Flask server** with CORS support
- **Endpoints**:
  - `/ping` → Health check
  - `/predict?place=<city>&date=<iso>` → Returns GeoJSON with predictions
- **Model loading**: Loads `cb_model.cbm` (CatBoost) from disk
- **Feature orchestration**: Coordinates all feature extraction steps

### Feature Engineering Modules

#### 1. Centrality Features (`feature_engineering/centrality_features.py`)
```python
compute_centrality(G, edges_gdf)
```
- Projects graph to metric CRS (EPSG:3857)
- Computes closeness & betweenness centrality
- Merges centralities back onto edge GeoDataFrame

#### 2. Time Features (`feature_engineering/time_features.py`)
```python
compute_time_features(edges_gdf, timestamp)
```
- Extracts Hour, is_weekend, time_of_day categories
- Attaches these columns to edges_gdf

#### 3. Land-use Features (`feature_engineering/landuse_features.py`)
```python
compute_landuse_edges(edges_gdf, land_gdf=None, allowed=None)
```
- Buffers each edge by BUFFER_METERS (150m)
- Picks nearest land-use polygon centroid among allowed types
- Fills "land_use" column on edges_gdf
- Default allowed types: {'residential', 'retail', 'commercial'}

#### 4. Highway Features (`feature_engineering/highway_features.py`)
```python
compute_highway(gdf, sensor_lookup=None)
```
- Looks up OSM "highway" tags per edge
- Handles list OSM IDs by normalizing to first element
- Falls back to spatial buffering if sensor_lookup provided
- Fills "highway" column on edges_gdf

## Machine Learning Overview

### Model Architecture
- **Algorithm**: CatBoost (Categorical Boosting)
- **Type**: Classification (5-class: volume bins 1-5)
- **Training Data**: Historical pedestrian sensor data from Melbourne, Sydney, and New York
- **Model File**: `cb_model.cbm` (serialized CatBoost model)

### Target Variable
- **Volume Bin**: 1-5 scale representing pedestrian volume levels
  - Bin 1: Very low pedestrian activity
  - Bin 2: Low pedestrian activity  
  - Bin 3: Moderate pedestrian activity
  - Bin 4: High pedestrian activity
  - Bin 5: Very high pedestrian activity

### Model Performance Metrics
- **Accuracy**: [To be measured on test set]
- **Precision/Recall**: Per-class performance
- **F1-Score**: Balanced measure for imbalanced classes
- **Cross-validation**: K-fold validation for robustness

## Feature Engineering Pipeline

### 1. Spatial Features

#### **Length Features**
```python
# Edge length in meters (projected to EPSG:3857)
length = edge_geometry.length
```
- **Purpose**: Longer streets typically have higher pedestrian volumes
- **Processing**: Project to metric CRS for accurate distance calculation

#### **Centrality Features**
```python
# Betweenness Centrality
betweenness = nx.betweenness_centrality(G, weight='length')

# Closeness Centrality  
closeness = nx.closeness_centrality(G, distance='length')
```
- **Betweenness**: Measures how often an edge lies on shortest paths
- **Closeness**: Measures average distance to all other nodes
- **Purpose**: Central streets attract more pedestrian traffic
- **Processing**: Computed on projected graph (EPSG:3857)

#### **Highway Classification**
```python
# OSM highway tags
highway_types = ['primary', 'secondary', 'tertiary', 'residential', 'footway', 'path']
```
- **Purpose**: Different street types have different pedestrian patterns
- **Processing**: 
  - Extract from OSM tags
  - Handle list OSM IDs by taking first element
  - Fallback to spatial buffering if needed

#### **Land-use Features**
```python
# Buffer-based land-use assignment
BUFFER_METERS = 150
allowed_landuse = {'residential', 'retail', 'commercial'}
```
- **Purpose**: Surrounding land-use affects pedestrian activity
- **Processing**:
  - Buffer each edge by 150 meters
  - Find nearest land-use polygon centroid
  - Assign dominant land-use type
  - Default to "other" if no match

### 2. Temporal Features

#### **Time-of-Day Categories**
```python
def categorize_time(hour):
    if 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 17: return 'afternoon' 
    elif 17 <= hour < 21: return 'evening'
    else: return 'night'
```
- **Purpose**: Pedestrian patterns vary significantly by time
- **Categories**: morning, afternoon, evening, night

#### **Weekend Indicator**
```python
is_weekend = timestamp.weekday() >= 5  # Saturday = 5, Sunday = 6
```
- **Purpose**: Weekend vs weekday patterns differ substantially

#### **Hour of Day**
```python
hour = timestamp.hour  # 0-23
```
- **Purpose**: Fine-grained temporal patterns within day

### 3. Feature Preprocessing

#### **Categorical Encoding**
```python
CAT_COLS = ["time_of_day", "land_use", "highway"]
# CatBoost handles categorical features natively
```
- **Method**: CatBoost's built-in categorical encoding
- **Advantage**: No need for one-hot encoding or label encoding

#### **Numerical Scaling**
```python
# Features that may need scaling:
# - length (meters)
# - betweenness (0-1 range)
# - closeness (0-1 range)
# - Hour (0-23)
```
- **Note**: CatBoost is tree-based, so scaling isn't strictly necessary
- **Recommendation**: Monitor feature importance for optimization

### 4. Feature Selection

#### **Final Feature Set**
```python
FEATS = [
    "length",           # Edge length in meters
    "betweenness",      # Betweenness centrality (0-1)
    "closeness",        # Closeness centrality (0-1)
    "Hour",            # Hour of day (0-23)
    "is_weekend",      # Boolean weekend flag
    "time_of_day",     # Categorical: morning/afternoon/evening/night
    "land_use",        # Categorical: residential/retail/commercial/other
    "highway",         # Categorical: primary/secondary/residential/etc
]
```

#### **Feature Importance Analysis**
- **Expected High Importance**: 
  - `time_of_day` (temporal patterns)
  - `land_use` (spatial context)
  - `betweenness` (network centrality)
- **Expected Medium Importance**:
  - `is_weekend` (temporal context)
  - `highway` (street type)
  - `length` (physical characteristics)
- **Expected Lower Importance**:
  - `closeness` (redundant with betweenness)
  - `Hour` (captured by time_of_day)

### 5. Data Quality Considerations

#### **Missing Data Handling**
- **OSM Data**: Graceful fallbacks for missing highway tags
- **Land-use**: Default to "other" category
- **Centrality**: Computed on available graph structure
- **Temporal**: Always available from request timestamp

#### **Outlier Detection**
- **Length**: Very long/short edges may indicate data issues
- **Centrality**: Extreme values may indicate graph artifacts
- **Recommendation**: Add outlier detection and handling

#### **Feature Validation**
```python
def validate_features(feats):
    assert feats['length'].min() > 0, "Negative lengths found"
    assert feats['betweenness'].between(0, 1).all(), "Invalid betweenness"
    assert feats['time_of_day'].isin(['morning','afternoon','evening','night']).all()
    # ... additional validations
```

### 6. Model Training Pipeline

#### **Data Sources**
- **Melbourne**: Primary training data (2019-2023)
- **Sydney**: Secondary training data
- **New York**: Additional training data
- **OSM**: Street network and attributes

#### **Training Process**
1. **Data Collection**: Historical sensor data + OSM networks
2. **Feature Extraction**: Apply same pipeline as inference
3. **Label Generation**: Convert sensor counts to volume bins
4. **Model Training**: CatBoost with cross-validation
5. **Hyperparameter Tuning**: Grid search or Bayesian optimization
6. **Model Validation**: Holdout test set evaluation
7. **Model Serialization**: Save as `cb_model.cbm`

#### **Training Configuration**
```python
# CatBoost parameters (example)
params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'random_seed': 42,
    'verbose': 100
}
```

### 7. Model Deployment Considerations

#### **Model Versioning**
- **Version Control**: Track model versions with metadata
- **A/B Testing**: Compare model performance in production
- **Rollback Strategy**: Ability to revert to previous model

#### **Model Monitoring**
- **Prediction Drift**: Monitor feature distributions
- **Performance Degradation**: Track accuracy over time
- **Data Quality**: Monitor input data quality

#### **Model Updates**
- **Retraining Schedule**: Regular model retraining
- **Incremental Learning**: Online learning for adaptation
- **Feature Engineering Updates**: New features or preprocessing

## Model Features

### Expected Feature Set (FEATS)
```python
FEATS = [
    "length",           # Edge length in meters
    "betweenness",      # Betweenness centrality
    "closeness",        # Closeness centrality  
    "Hour",            # Hour of day (0-23)
    "is_weekend",      # Boolean weekend flag
    "time_of_day",     # Categorical: morning/afternoon/evening/night
    "land_use",        # Categorical: residential/retail/commercial/other
    "highway",         # Categorical: primary/secondary/residential/etc
]
```

### Categorical Features (CAT_COLS)
```python
CAT_COLS = ["time_of_day", "land_use", "highway"]
```

## Current Status

### ✅ Completed
- Flask server starts successfully
- `/ping` endpoint works
- Centrality features integrate successfully
- Temporal features integrate successfully
- Land-use features integrate successfully
- Highway features integrate successfully (fixed list OSM ID issue)
- Model loading and prediction pipeline

### 🔧 Recent Fixes
- **Fixed land-use import**: Changed from `assign_landuse_by_nearest` to `compute_landuse_edges`
- **Fixed highway merge error**: Added OSM ID normalization to handle list values
- **Added debug logging**: Each feature step logs shape information

### 🚧 Current Issues
- Need to test full prediction pipeline end-to-end
- Land-use data path may need adjustment for different cities
- Performance optimization needed for large cities

## Data Dependencies

### Required Files
- `cb_model.cbm` - Pre-trained CatBoost model
- Land-use polygons (optional, falls back to "other" if missing)

### OSM Data
- Automatically downloaded via osmnx for any city
- Network type: "walk" (pedestrian network)
- Includes highway tags and geometry

## API Usage Examples

### Basic Prediction
```bash
curl "http://localhost:5000/predict?place=Tel%20Aviv"
```

### Prediction with Date
```bash
curl "http://localhost:5000/predict?place=Melbourne&date=2023-12-25T14:30:00"
```

### Health Check
```bash
curl "http://localhost:5000/ping"
```

## Response Format

The API returns a GeoJSON FeatureCollection where each feature represents a street segment with:
- **geometry**: LineString of the street segment
- **properties**: 
  - `volume_bin`: Predicted pedestrian volume (1-5)
  - `osmid`: OSM ID(s) of the segment
  - `length`: Segment length in meters
  - `highway`: Highway classification
  - `land_use`: Land-use classification
  - All other OSM attributes

## Performance Considerations

### Current Bottlenecks
1. **OSM download**: Each request downloads fresh OSM data
2. **Centrality computation**: O(n²) complexity for large graphs
3. **Land-use buffering**: Spatial operations on all edges
4. **Model prediction**: Sequential processing

### Optimization Opportunities
1. **Cache OSM graphs** by city
2. **Parallelize centrality** computation
3. **Pre-compute land-use** assignments
4. **Batch predictions** for multiple timestamps
5. **GPU acceleration** for CatBoost (if available)

## Future Development Goals

### Phase 1: Production Readiness
- [ ] Add comprehensive error handling
- [ ] Implement request validation
- [ ] Add API rate limiting
- [ ] Deploy behind Gunicorn + Nginx
- [ ] Add monitoring and logging

### Phase 2: Performance Optimization
- [ ] Implement OSM graph caching
- [ ] Parallelize feature extraction
- [ ] Add Redis for session storage
- [ ] Optimize spatial operations
- [ ] Add model versioning

### Phase 3: Feature Expansion
- [ ] Support for multiple cities simultaneously
- [ ] Real-time sensor data integration
- [ ] Weather data integration
- [ ] Event/calendar data integration
- [ ] Historical trend analysis

### Phase 4: Advanced Features
- [ ] Confidence intervals for predictions
- [ ] Anomaly detection
- [ ] Seasonal adjustments
- [ ] Multi-modal transportation integration
- [ ] Mobile app integration

## Deployment Architecture

### Development
```bash
cd pedestrian-api
python app.py
```

### Production (Recommended)
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# With Nginx reverse proxy
# (nginx config needed)
```

### Docker (Future)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Testing Strategy

### Unit Tests
- [ ] Feature engineering functions
- [ ] Model prediction accuracy
- [ ] API endpoint responses

### Integration Tests
- [ ] End-to-end prediction pipeline
- [ ] Different city scenarios
- [ ] Error handling scenarios

### Performance Tests
- [ ] Large city processing time
- [ ] Concurrent request handling
- [ ] Memory usage profiling

## Monitoring and Observability

### Metrics to Track
- Request latency
- Prediction accuracy
- Error rates
- Resource usage
- Cache hit rates

### Logging
- Request/response logs
- Feature extraction timing
- Model prediction timing
- Error stack traces

## Security Considerations

### API Security
- Input validation and sanitization
- Rate limiting
- CORS configuration
- Error message sanitization

### Data Security
- Secure model storage
- Encrypted data transmission
- Access control for sensitive data

## Documentation

### API Documentation
- OpenAPI/Swagger specification
- Example requests/responses
- Error code documentation

### Developer Documentation
- Setup instructions
- Architecture diagrams
- Contributing guidelines

---

**Last Updated**: July 2025
**Version**: 1.0.0
**Status**: Development Phase 