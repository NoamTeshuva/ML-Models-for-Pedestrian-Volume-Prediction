# Dynamic Land Use GeoPackage for Pedestrian Volume API

## Overview

The pedestrian volume API now supports dynamic land use generation for any city or bounding box, eliminating the dependency on precomputed GeoPackage files.

## Features

✅ **Dynamic Land Use Generation**: Automatically fetches land use data from OpenStreetMap for any place or bounding box  
✅ **Intelligent Caching**: Saves downloaded data to avoid repeated API calls  
✅ **Fallback Support**: Gracefully handles cases where no land use data is available  
✅ **Automatic Cleanup**: Removes old temporary files to save disk space  
✅ **Multiple Input Methods**: Supports both place names and bounding boxes  

## API Usage

### Place-based Prediction
```bash
curl "http://localhost:5000/predict?place=Monaco&date=2024-01-15"
```

### Bounding Box Prediction
```bash
curl "http://localhost:5000/predict?bbox=7.4,43.7,7.5,43.8&date=2024-01-15"
```

### Combined Approach
```bash
curl "http://localhost:5000/predict?place=Monaco&bbox=7.4,43.7,7.5,43.8&date=2024-01-15"
```

## Implementation Details

### Core Functions

#### `get_landuse_polygons(place=None, bbox=None, save_path=None)`
- **Purpose**: Dynamically fetch and construct land use polygon layer
- **Parameters**:
  - `place`: Place name (e.g., "Monaco", "Melbourne, Australia")
  - `bbox`: Bounding box as (minx, miny, maxx, maxy)
  - `save_path`: Optional custom save path
- **Returns**: GeoDataFrame with land use polygons

#### `compute_landuse_edges(edges_gdf, place=None, bbox=None, ...)`
- **Purpose**: Add land use classification to street edges
- **Parameters**: 
  - `edges_gdf`: Street network edges
  - `place`/`bbox`: For dynamic land use generation
- **Returns**: Edges with added 'land_use' column

#### `cleanup_temp_files(max_age_hours=24)`
- **Purpose**: Remove old temporary files
- **Parameters**: `max_age_hours`: Maximum age before deletion

### Caching Strategy

1. **Cache Key Generation**: Uses MD5 hash of place name or bbox coordinates
2. **File Naming**: `{place}_{hash}_landuse.gpkg` or `bbox_{hash}_landuse.gpkg`
3. **Storage Location**: `pedestrian-api/temp/` directory
4. **Cache Hit**: Loads existing file if available
5. **Cache Miss**: Downloads from OSM and saves to disk

### Land Use Categories

The system filters for these relevant categories:
- `residential`
- `commercial` 
- `retail`
- `industrial`
- `recreation_ground`
- `park`

### Error Handling

- **No Place Found**: Falls back to bbox if provided
- **No Land Use Data**: Creates empty GeoDataFrame, assigns "other" to all edges
- **Network Errors**: Logs warnings and continues with fallback options

## Testing

Run the test script to verify functionality:

```bash
cd pedestrian-api
python test_dynamic_landuse.py
```

This will test:
- Place-based land use generation
- Bbox-based land use generation  
- Caching functionality
- Edge case handling
- Cleanup operations

## File Structure

```
pedestrian-api/
├── app.py                           # Updated Flask API
├── feature_engineering/
│   └── landuse_features.py         # Enhanced with dynamic support
├── temp/                           # Cache directory (auto-created)
│   ├── Monaco_abc12345_landuse.gpkg
│   └── bbox_def67890_landuse.gpkg
├── test_dynamic_landuse.py         # Test script
└── DYNAMIC_LANDUSE_README.md       # This file
```

## Performance Considerations

- **First Request**: May take 10-30 seconds to download OSM data
- **Subsequent Requests**: Near-instantaneous due to caching
- **Memory Usage**: Land use data is loaded only when needed
- **Disk Usage**: Temporary files are automatically cleaned up

## Migration from Static Files

The API maintains backward compatibility:
- If no `place` or `bbox` is provided, falls back to original static file
- Existing code continues to work without modification
- Gradual migration possible

## Future Enhancements

- Redis-based caching for distributed deployments
- Configurable land use categories
- Batch processing for multiple cities
- Integration with other OSM data sources