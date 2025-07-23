import numpy as np
import pandas as pd

NUM_LOCATIONS = 3
SENSORS_PER_LOCATION = 10
NG_SENSORS_PER_LOCATION = 5
OK_SENSORS_PER_LOCATION = SENSORS_PER_LOCATION - NG_SENSORS_PER_LOCATION
NUM_DATAPOINTS = 1000
LOCATIONS = ['A', 'B', 'C']

def generate_sensor_data():
    """
    Generates sample sensor data with location_id.

    Returns:
        pandas.DataFrame: DataFrame with columns 'location_id', 'sensor_id',
                          'timestamp', 'value', and 'result'.
    """
    data = []
    sensor_counter = 1
    for loc_idx, location_id in enumerate(LOCATIONS):
        for i in range(SENSORS_PER_LOCATION):
            # Unique sensor ID across all locations
            sensor_id = f"sensor_{sensor_counter:02d}"
            sensor_counter += 1

            timestamps = np.arange(NUM_DATAPOINTS)
            # Generate some waveform data (e.g., sine wave with noise)
            # Vary frequency per sensor based on overall sensor index
            base_wave = np.sin(timestamps / (50 + (loc_idx * SENSORS_PER_LOCATION + i) * 2))
            noise = np.random.normal(0, 0.2, NUM_DATAPOINTS)
            values = base_wave + noise

            # Assign results (OK/NG) per location
            if i < NG_SENSORS_PER_LOCATION:
                result = "NG"
            else:
                result = "OK"

            for t, v in zip(timestamps, values):
                data.append([location_id, sensor_id, t, v, result])

    df = pd.DataFrame(data, columns=['location_id', 'sensor_id', 'timestamp', 'value', 'result'])
    return df

if __name__ == '__main__':
    sensor_df = generate_sensor_data()
    print(f"Generated DataFrame shape: {sensor_df.shape}")
    print(sensor_df.head())

    print("\nResult distribution per location:")
    print(sensor_df.groupby(['location_id', 'sensor_id'])['result'].first().groupby('location_id').value_counts())

    print("\nOverall result distribution for sensors:")
    print(sensor_df.groupby('sensor_id')['result'].first().value_counts())

    # Verify counts
    total_sensors = len(LOCATIONS) * SENSORS_PER_LOCATION
    print(f"\nTotal sensors generated: {sensor_df['sensor_id'].nunique()} (Expected: {total_sensors})")

    for loc in LOCATIONS:
        loc_df = sensor_df[sensor_df['location_id'] == loc]
        ng_count = loc_df.groupby('sensor_id')['result'].first().value_counts().get('NG', 0)
        ok_count = loc_df.groupby('sensor_id')['result'].first().value_counts().get('OK', 0)
        print(f"Location {loc}: NG sensors = {ng_count} (Expected: {NG_SENSORS_PER_LOCATION}), OK sensors = {ok_count} (Expected: {OK_SENSORS_PER_LOCATION})")

    # Save to a CSV for inspection if needed (optional)
    # sensor_df.to_csv("sample_sensor_data_with_location.csv", index=False)
    # print("\nSample data saved to sample_sensor_data_with_location.csv")
