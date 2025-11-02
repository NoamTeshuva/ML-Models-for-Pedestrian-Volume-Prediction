"""
YAML configuration utility for v3 feature extraction pipelines.
Loads config/cities.yaml if present and merges defaults with city-specific overrides.
"""
import os
import yaml
from typing import Dict, Any, Optional


def load_v3_config(city: str = "melbourne") -> Dict[str, Any]:
    """
    Load v3 configuration parameters for a city from config/cities.yaml.
    
    Args:
        city: City name to load config for
        
    Returns:
        Dict with merged defaults and city-specific v3 parameters
    """
    # Safe defaults for v3 extractors
    defaults = {
        "max_seg_m": 20,
        "smooth_window": 3,
        "dem_nodata": -32768,
        "min_valid_fraction": 0.2,
        "buffer_m": 25,
        "ndvi_threshold": 0.25,
        "ndvi_rescale": "auto",
        "treat_zero_as_nodata": True,
        "min_coverage": 0.3
    }
    
    config_path = "config/cities.yaml"
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using defaults only")
        return defaults
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # Merge city-specific v3 overrides if present
        city_config = config.get("cities", {}).get(city, {}).get("v3", {})
        merged = defaults.copy()
        merged.update(city_config)
        
        print(f"Loaded config for {city}: {len(city_config)} overrides from {config_path}")
        return merged
        
    except Exception as e:
        print(f"Warning: Failed to load {config_path}: {e}")
        print("Using safe defaults only")
        return defaults


def get_topo_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract topography-specific parameters from merged config."""
    return {
        "max_seg_m": config.get("max_seg_m", 20),
        "smooth_window": config.get("smooth_window", 3),
        "dem_nodata": config.get("dem_nodata", -32768),
        "min_valid_fraction": config.get("min_valid_fraction", 0.2)
    }


def get_canopy_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract canopy-specific parameters from merged config."""
    return {
        "buffer_m": config.get("buffer_m", 25),
        "ndvi_threshold": config.get("ndvi_threshold", 0.25),
        "ndvi_rescale": config.get("ndvi_rescale", "auto"),
        "treat_zero_as_nodata": config.get("treat_zero_as_nodata", True),
        "min_valid_fraction": config.get("min_valid_fraction", 0.2)
    }