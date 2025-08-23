#!/bin/bash
# Development server startup script for Israel OSM Network API

echo "Starting Israel OSM Network API server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload