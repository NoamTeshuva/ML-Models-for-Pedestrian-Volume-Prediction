import pandas as pd

# File path from your example
file_path = r'C:\Users\HP\PycharmProjects\Pedestrian_Volume\data\raw\melbourne\January_2019.csv'
df = pd.read_csv(file_path)

# Combine 'Date' and 'Hour' into a datetime column
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Hour'].astype(str) + ':00:00', format='%d/%m/%Y %H:%M:%S')

# Day Type: Weekday (0–4) or Weekend (5–6)
df['day_type'] = df['datetime'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Season (Southern Hemisphere logic)
def get_season(month):
    if month in [9, 10, 11]:
        return 'Spring'
    elif month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'

df['season'] = df['datetime'].dt.month.apply(get_season)

# Time of Day classification
def get_time_of_day(hour):
    if 6 <= hour <= 10:
        return 'Morning'
    elif 11 <= hour <= 14:
        return 'Noon'
    elif 16 <= hour <= 19:
        return 'Afternoon'
    elif 19 <= hour <= 23:
        return 'Night'
    else:
        return 'Other'  # Early morning hours like 0–5

df['time_of_day'] = df['datetime'].dt.hour.apply(get_time_of_day)

# Optional: Save the processed CSV
output_path = r'C:\Users\HP\PycharmProjects\Pedestrian_Volume\data\processed\melbourne_time_features.csv'
df.to_csv(output_path, index=False)

print("✅ Time features created and saved successfully!")
