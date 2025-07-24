import numpy as np
import pandas as pd
import datetime

def generate_sensor_data():
    """
    Generates a larger sample of sensor data (approx. 30 unique AssyNo).
    Returns a DataFrame with original column names.
    """
    data = []
    base_time = datetime.datetime(2025, 5, 16, 13, 30, 38)
    num_samples = 30

    # Generate lists of data for 30 samples
    assy_nos = [f"JP4466104371250516{i:04d}" for i in range(num_samples)]

    # Create more diverse hinbans
    hinban_bases = ["4466104371", "4466104372", "4466105510", "4466105511"]
    hinbans = [np.random.choice(hinban_bases) for _ in range(num_samples)]

    # Create more diverse ng_types
    ng_type_options = ["チューブ漏れ", "カシメ不良", "異音", "センサーずれ", "OK"] # Add "OK" for non-ng items
    # Skew the distribution: more OK, some common failures, few rare ones
    ng_type_probabilities = [0.25, 0.25, 0.1, 0.1, 0.3]
    ng_types = np.random.choice(ng_type_options, num_samples, p=ng_type_probabilities)

    # Create results based on ng_type
    results = ["OK" if ng == "OK" else "NG" for ng in ng_types]

    # Create diverse sensor types
    sensor_type_options = [f"SNS{i}[{j}]" for i in range(1, 5) for j in range(4)]
    sensor_types = [np.random.choice(sensor_type_options) for _ in range(num_samples)]


    for i in range(num_samples):
        num_datapoints = np.random.randint(200, 500)
        timestamps = [base_time + datetime.timedelta(seconds=j) for j in range(num_datapoints)]

        # Generate waveform data - make NG waveforms slightly different
        base_amplitude = 50
        noise_level = 5
        if results[i] == "NG":
            # Introduce some anomaly for NG parts
            base_amplitude = np.random.uniform(55, 70)
            noise_level = np.random.uniform(7, 12)

        base_wave = np.sin(np.linspace(0, np.random.choice([5, 10, 15]) * np.pi, num_datapoints)) * base_amplitude + 80
        noise = np.random.normal(0, noise_level, num_datapoints)
        sensor_values = base_wave + noise

        # Add a spike anomaly for a specific NG type
        if ng_types[i] == "カシメ不良":
             spike_start = num_datapoints // 2
             sensor_values[spike_start : spike_start + 5] += 30

        for j in range(num_datapoints):
            data.append([
                timestamps[j],
                assy_nos[i],
                0, 10,  # CoreAssy1[10], CoreAssy1[42]
                hinbans[i],
                j + 1,  # No
                results[i],
                sensor_types[i],
                sensor_values[j],
                ng_types[i]
            ])

    df = pd.DataFrame(data, columns=[
        'DATE_AND_TIME', 'AssyNo', 'CoreAssy1[10]', 'CoreAssy1[42]',
        'hinban', 'No', 'result', 'Sensor_Type', 'Sensor_Value', 'ng_type'
    ])
    return df

if __name__ == '__main__':
    sensor_df = generate_sensor_data()
    print(f"Generated DataFrame shape: {sensor_df.shape}")
    print(f"Number of unique AssyNo: {sensor_df['AssyNo'].nunique()}")
    print("\nColumns:", sensor_df.columns)
    print("\nResult distribution:\n", sensor_df.groupby('AssyNo')['result'].first().value_counts())
    print("\nNG Type distribution:\n", sensor_df.groupby('AssyNo')['ng_type'].first().value_counts())
    print("\nHinban distribution:\n", sensor_df.groupby('AssyNo')['hinban'].first().value_counts())
    print("\nSample Data Head:")
    print(sensor_df.head())
