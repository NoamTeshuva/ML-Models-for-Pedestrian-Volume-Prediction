# Scripts Directory

All Python scripts organized by purpose.

## Directory Structure

### `model_training/`
Scripts for training multi-city pedestrian volume prediction models.

**Key Scripts:**
- `train_3city_with_weather.py` - **MAIN PRODUCTION MODEL** - Trains on Melbourne + Dublin + Zurich with weather features
- `train_mel_dub_test_zurich.py` - **LEAVE-ONE-CITY-OUT TEST** - Tests generalization to unseen cities
- `train_3city_mel_dub_zur_cuda.py` - Trains without weather features (baseline)
- `train_3city_model_cuda.py` - Original 3-city model
- `train_with_time_of_day_cuda.py` - Alternative model using time_of_day instead of Hour

**Results:**
- Production model (with weather): **74.44% accuracy**
- Leave-one-city-out (Mel+Dub → Zurich): **38.58% accuracy**

### `tel_aviv/`
Scripts for Tel Aviv data processing and validation attempts.

**Scripts:**
- `add_temporal_features_tlv.py` - Add time features to Tel Aviv data
- `analyze_tel_aviv_ground_truth.py` - Analyze ground truth data structure
- `map_tlv_sensors_to_canopy.py` - Map sensors to green canopy features
- `merge_tlv_final_features.py` - Merge all features for Tel Aviv
- `merge_tlv_ground_truth_with_features.py` - Attempt to merge ground truth
- `validate_model_on_tel_aviv.py` - Validate model on Tel Aviv ground truth
- `validate_tlv_time_of_day.py` - Alternative validation approach
- `validate_with_merged_data.py` - Merged data validation
- `visualize_tlv_predictions.py` - Visualize predictions

**Status:** Tel Aviv ground truth incompatible with our approach (different granularity/features)

### `zurich/`
Scripts for Zurich data processing and feature extraction.

**Scripts:**
- `complete_zurich_dataset.py` - Complete Zurich feature extraction pipeline
- `analyze_zurich_features.py` - Analyze Zurich feature quality

**Results:** Successfully processed 38,509 records with all features

### `archive/`
Old/deprecated scripts kept for reference.

Currently empty - move unused scripts here as needed.

---

## Usage

### To Train Production Model:
```bash
cd scripts/model_training
python train_3city_with_weather.py
```

### To Test Generalization to New City:
```bash
cd scripts/model_training
python train_mel_dub_test_zurich.py
```

---

## Model Files

Trained models are saved in `pedestrian-api/`:
- `cb_model_3city_with_weather.cbm` - **PRODUCTION MODEL** (74.44% accuracy)
- `cb_model_3city_mel_dub_zur.cbm` - Baseline without weather (74.16% accuracy)

---

## Key Findings

1. **Weather features add minimal value** (+0.28% accuracy)
   - Temperature: 0.97% importance
   - Precipitation: 0.24% importance
   - Wind: 0.15% importance

2. **Hour is most important feature** (22.54% importance)

3. **Generalization is limited**:
   - 74.44% accuracy on cities in training
   - Only 38.58% on completely unseen city (Zurich)

4. **To deploy on new city**: Need to collect sample data and retrain

---

## Documentation

See root directory for detailed documentation:
- `WEATHER_MODEL_COMPARISON.md` - Weather vs no-weather comparison
- `TRAIN_TEST_SPLIT_EXPLAINED.md` - How train/test split works
