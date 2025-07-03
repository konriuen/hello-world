import numpy as np
import pandas as pd

NUM_SENSORS = 30
NUM_DATAPOINTS = 1000

def generate_sensor_data():
    """
    Generates sample sensor data.

    Returns:
        pandas.DataFrame: DataFrame with columns 'sensor_id', 'timestamp', 'value', and 'result'.
    """
    data = []
    for i in range(NUM_SENSORS):
        sensor_id = f"sensor_{i+1:02d}"
        timestamps = np.arange(NUM_DATAPOINTS)
        # Generate some waveform data (e.g., sine wave with noise)
        base_wave = np.sin(timestamps / (50 + i*5)) # Vary frequency per sensor
        noise = np.random.normal(0, 0.2, NUM_DATAPOINTS)
        values = base_wave + noise

        # Assign results (OK/NG)
        # Ensure roughly half are NG
        if i < NUM_SENSORS / 2:
            result = "NG"
        else:
            result = "OK"

        # For simplicity in this example, we'll make all data points for a given NG sensor also NG.
        # In a real scenario, this might be more complex (e.g. result applies to the whole waveform).
        # However, for individual point coloring, we need a result per point or per waveform.
        # Let's assume the 'result' applies to the entire waveform for now.
        # We will store it alongside each data point for easier access in Plotly.

        for t, v in zip(timestamps, values):
            data.append([sensor_id, t, v, result])

    df = pd.DataFrame(data, columns=['sensor_id', 'timestamp', 'value', 'result'])
    return df

if __name__ == '__main__':
    sensor_df = generate_sensor_data()
    print(f"Generated DataFrame shape: {sensor_df.shape}")
    print(sensor_df.head())
    print("\nResult distribution:")
    print(sensor_df.groupby('sensor_id')['result'].first().value_counts())

    # Check if 'result' column has roughly half NG and half OK for sensors
    sensor_results = sensor_df.groupby('sensor_id')['result'].first()
    ng_count = (sensor_results == 'NG').sum()
    ok_count = (sensor_results == 'OK').sum()
    print(f"\nNumber of NG sensors: {ng_count}")
    print(f"Number of OK sensors: {ok_count}")

    # Save to a CSV for inspection if needed (optional)
    # sensor_df.to_csv("sample_sensor_data.csv", index=False)
    # print("\nSample data saved to sample_sensor_data.csv")
