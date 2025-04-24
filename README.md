# Pedestrian Volume Prediction Project

## Overview

This project focuses on developing machine learning models for predicting pedestrian volume at the street-segment level across various global cities. The primary objective is to estimate pedestrian volume in areas lacking local sensor data by leveraging transfer learning techniques combined with street network features extracted from OpenStreetMap (OSM).

The research is conducted as part of an initiative at the **Civil Engineering Department** of Ariel University and is currently in the **initial data collection and preprocessing phase**.

---

## Data Sources and Description

The project integrates pedestrian count datasets from multiple global cities. These datasets vary by measurement periods, collection methods, and metadata availability. The key characteristics of each data source are summarized below:

### Toulouse, France
- **Measurement Period:** 2006–2019  
- **Days:** Tuesdays and Saturdays  
- **Times:** Morning (07:30–09:00), Afternoon (11:30–12:30), Evening (16:30–18:30)  
- **Data:** Point locations, street names, time, day, year  
- **Street Type:** Not provided (classification required via cross-referencing)  
- **Climate:** Not specified  
- **Measurement Method:** Exact pedestrian counts  

### Sydney, Australia
- **Measurement Period:** Starting 2020, ongoing  
- **Data:** Location ID, place name, date, pedestrian counts, moving average metrics  
- **Street Type:** Names provided, no further description  
- **Climate:** Not included  
- **Measurement Method:** Exact pedestrian counts  

### Melbourne, Australia
- **Measurement Period:** 2009–present (monthly data from 2013)  
- **Data:** Hourly pedestrian counts per sensor, 24-hour coverage  
- **Street Type:** Not available  
- **Climate:** Not specified  
- **Measurement Method:** Sensor-based pedestrian counts  

### New York, USA
- **Measurement Period:** Selected dates in May and September  
- **Days:** Weekdays (07:00–09:00, 16:00–19:00), Saturdays (12:00–14:00)  
- **Data:** Street/block ID, day, month, year  
- **Street Type:** Includes commercial streets, bridges, pedestrian bridges  
- **Climate:** Not specified  
- **Measurement Method:** Requires clarification (likely from ArcGIS database)  

### BikePed Portal (Primarily Portland, USA)
- **Measurement Period:** 2025–present *(check for potential typo)*  
- **Data:** Geolocation, city, district, country, movement direction, metadata, weather conditions, event markers  
- **Street Type:** Main roads, highways, side streets, sidewalks, crossings  
- **Climate:** Included (weather conditions, severe weather, events)  
- **Measurement Method:** Various sensor types, updated every 15 minutes  

### Delaware, USA
- **Measurement Period:** 2014–2022  
- **Data:** 24/7 pedestrian and cyclist counts, infrared sensors  
- **Street Type:** Intersections, urban and rural areas  
- **Climate:** Not referenced  
- **Measurement Method:** Infrared sensors, supplemental survey data  

### Zurich, Switzerland
- **Measurement Period:** 2019–2023  
- **Data:** 15-minute intervals (96 periods per day), geolocation, pedestrian and bicycle counts  
- **Street Type:** Main roads, side streets, city entrances, neighborhood connectors  
- **Climate:** Not specified  
- **Measurement Method:** Infrared sensors  

### Dublin, Ireland
- **Data Quality:** Limited (127 rows), requires further validation  
- **Consideration:** Check periods for traffic restrictions (e.g., COVID-19 lockdowns)  

---

## Planned Feature Extraction

The following features will be extracted from OpenStreetMap and used for model training and inference:

- **Street Centrality Measures:** Closeness, betweenness (via `osmnx` library)  
- **Highway Type:** Classification of streets (e.g., primary, secondary, residential)  
- **Land Use Features:** Extracted where available from OSM  

### Temporal Features
- Hour of the day (1–24)  
- Seasons (spring, summer, autumn, winter)  
- Day type (weekday/weekend)  
- Time categories (morning, noon, evening, night)  

The raw data will be standardized to a **maximum temporal resolution of one hour**.

---

## Data Processing Plan

1. **Select Target Dataset:**  
   Focus initially on Sydney (2020–2021), extracting one year of data.

2. **Feature Engineering:**  
   Integrate street network features with pedestrian count data.

3. **Standardization:**  
   Convert all datasets into a unified schema, preferably **GeoPackage format**.

4. **Model Training Options:**  
   - Train a separate model for each city  
   - Merge datasets to create a global model  
   - Apply transfer learning to adapt models between cities  

---

## Current Status

The project is currently in the **data preparation and feature extraction phase**. Modeling and evaluation stages will commence after preprocessing is completed.

---

## Contact

**Noam Teshuva**  
CS Candidate, Lab Assistant  
Civil Engineering Department, Ariel University  
Email: [teshvau91@gmail.com](mailto:teshvau91@gmail.com)

**Supervisor:** Prof. Achituv Cohen
