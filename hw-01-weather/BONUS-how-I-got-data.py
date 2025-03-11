from datetime import datetime
from meteostat import Hourly, Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to download weather data
def download_weather_data(lat, lon, start, end):
    """Downloads and filters weather data for the given latitude, longitude, and time period.
    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        start (str): Start date in DD-MM-YYYY format.
        end (str): End date in DD-MM-YYYY format.
    Returns:
        pd.DataFrame: Filtered DataFrame with temp, pres, and coco columns.
    """
    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start, '%d-%m-%Y')
    end_date = datetime.strptime(end, '%d-%m-%Y')
    # Create location point
    location = Point(lat, lon)
    # Fetch hourly data
    data = Hourly(location, start_date, end_date).fetch()
    # Reset index to include time as a column
    data.reset_index(inplace=True)
    return data

# Function to process weather data
def process_weather_data(data):
    # Ensure all values exist by replacing NaNs with 0.0
    data = data.fillna(0.0)

    # Category mapping for 'coco'
    category_mapping = {
        1: 1, 2: 1,  # Clear
        3: 2, 4: 2,  # Cloudy
        5: 3, 6: 3,  # Fog
        7: 4, 8: 4, 9: 4,  # Rain
        10: 5, 11: 5,  # Freezing Rain
        12: 6, 13: 6,  # Sleet
        14: 7, 15: 7, 16: 7,  # Snowfall
        17: 8, 18: 8,  # Rain Shower
        19: 9, 20: 9,  # Sleet Shower
        21: 10, 22: 10,  # Snow Shower
        23: 11, 25: 11, 26: 11, 27: 11,  # Storm
        24: 12  # Hail
    }

    # Apply category mapping to 'coco' column after converting it to int
    data['coco'] = data['coco'].astype(int).map(category_mapping)

    return data

# Function to split and save data
def split_and_save_data(data):
    # Shuffle the data
    data_shuffled = data.sample(frac=1).reset_index(drop=True)

    # Split data into train (60%) and test sets (40%)
    train_size = int(0.6 * len(data_shuffled))
    train_data, test_data = data_shuffled[:train_size], data_shuffled[train_size:]

    # Save train data
    train_data.to_csv('data/train-data.csv', index=False)

    # Save test data without the last column ('coco')
    test_data_without_targets = test_data.iloc[:, :-1]
    test_data_without_targets.to_csv('data/test-data-extra.csv', index=False)

    # Save test targets
    test_targets = test_data['coco'].tolist()
    with open('data/test-data-targets-extra.csv', 'w') as f:
        for target in test_targets:
            f.write(str(target) + '\n')

# Download data
# I won't tell you what the lat and lon I used and what the start and end dates are, but I will tell you that it was somewhere in Romania
# The lat and lon and dates you see here are just placeholders so it will run
weather_data = download_weather_data(lat=49.7283, lon=13.3478, start='01-01-2023', end='01-11-2024')
import os
os.makedirs('data', exist_ok=True)
# Save raw weather data to CSV
weather_data.to_csv('data/raw-weather-data-extra.csv', index=False)

# Process data
processed_data = process_weather_data(weather_data)

# Split and save processed data
split_and_save_data(processed_data)

# Plot distribution of labels
test_targets = pd.read_csv('data/test-data-targets-extra.csv', header=None)
plt.figure(figsize=(10, 6))
plt.bar(test_targets[0].value_counts().index, test_targets[0].value_counts())
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()