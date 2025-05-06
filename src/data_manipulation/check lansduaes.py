import geopandas as gpd

# load your new GPKG
edges = gpd.read_file(
    r"C:\Users\HP\PycharmProjects\Pedestrian_Volume\data\processed\melbourne\melbourne_landuse_features.gpkg",
    layer="edges",
)

# 1) Show a few rows
print(edges[["osmid","landuse"]].head(10))

# 2) Count how many segments got each land-use (and how many remain NaN)
print(edges["landuse"].value_counts(dropna=False))
