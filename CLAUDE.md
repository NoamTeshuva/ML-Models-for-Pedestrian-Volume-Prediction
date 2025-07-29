# CLAUDE.md — System Prompt for Claude Code

## 🧠 Project Context
You are assisting with a machine learning project that predicts pedestrian volume in any city based on OpenStreetMap features and time data.

## 📂 Project Structure
- `feature_engineering/`: Contains modules like `landuse_features.py`, `centrality_features.py`
- `data/processed/`: Stores output `.csv` and `.gpkg` files
- `app.py`: Flask API entrypoint
- Dynamic GeoPackage files are stored under `temp/`

## 📦 Libraries Used
- `osmnx`, `geopandas`, `fiona`, `shapely`, `pyproj`
- `scikit-learn`, `catboost`, `pandas`, `numpy`
- `Flask` for API

## 🌍 Cities Supported
Supports any city in the world via dynamic OSM queries (`place` or `bbox`)

## 🧱 Coding Guidelines
- Write **modular** functions, no hardcoded city names or paths
- Always use `os.makedirs(..., exist_ok=True)` before saving files
- Prefer **functional composition** over monolithic code
- Include type hints and docstrings
- Add `try/except` blocks for file-not-found and empty OSM results
- Always return GeoDataFrames or DataFrames with defined schemas

## 📜 Dynamic Feature Extraction
When user requests prediction for a city:
- Download land use polygons with `osmnx`
- Save to `temp/{place}_landuse.gpkg`
- Use cached file if it already exists
- Clean up temp files > 6 hours old (optional)

## ⚡ Performance Considerations
- Cache expensive OSM queries for at least 1 hour
- Use spatial indexing for large datasets
- Implement request rate limiting for API endpoints
- Batch process multiple requests when possible
- Use memory-efficient operations for large GeoDataFrames

## 🚨 Error Handling
- Return consistent error JSON: `{"error": "message", "code": 400, "details": null}`
- Log all exceptions with city context for debugging
- Provide meaningful error messages for common failures (OSM timeout, invalid place name)
- Implement graceful degradation when optional features fail

## 🔍 Input Validation
- Validate `place` names exist in OSM before processing
- Check `bbox` coordinates are within valid lat/lng ranges (-90 to 90 lat, -180 to 180 lng)
- Ensure `date` is parseable and reasonable (not too far in past/future)
- Sanitize all user inputs to prevent injection attacks
- Validate required fields are present and non-empty

## ✅ API Rules
- `/predict` endpoint must accept `place`, `bbox`, and `date`
- Return a structured JSON response with model prediction
- All endpoints must handle bad input gracefully and log errors
- Include response time and cache status in API responses
- Implement proper HTTP status codes (200, 400, 404, 500, 503)

## 🧪 Testing
- Add test files like `test_dynamic_landuse.py`
- Validate all returned GeoDataFrames with `.empty` check
- Unit test the feature engineering layers separately
- Test edge cases: empty OSM results, invalid coordinates, network timeouts
- Include integration tests for full API workflow

## 📋 Communication Rules
- **Always provide a 2-line summary** before any code changes or suggestions
- Line 1: Briefly explain what task you're about to perform
- Line 2: Describe the approach or method you'll use
- This ensures clear communication and allows the user to understand your intent before execution

## ✅ Claude Behavior
- Always suggest code changes as diffs or patch-style edits
- When unsure, ask for clarification
- When implementing new logic, first create a stub, then fill it
- Summarize and comment any logic that involves geospatial joins, buffers, or centrality
- Prioritize code readability and maintainability over cleverness

---
Use this file to automatically set Claude's coding rules and context.