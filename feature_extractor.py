import numpy as np
import pandas as pd

# --- Basic Statistical Features ---
def calculate_mean(series):
    """Calculates the mean of a series."""
    return np.mean(series)

def calculate_std(series):
    """Calculates the standard deviation of a series."""
    return np.std(series)

def calculate_max(series):
    """Calculates the maximum value of a series."""
    return np.max(series)

def calculate_min(series):
    """Calculates the minimum value of a series."""
    return np.min(series)

def calculate_rms(series):
    """Calculates the Root Mean Square (RMS) of a series."""
    return np.sqrt(np.mean(series**2))

# --- Segmented Features ---
def calculate_segmented_features(series, n_segments):
    """
    Splits the series into n_segments and calculates mean, max, min for each segment.

    Args:
        series (pd.Series or np.array): The input time series data.
        n_segments (int): The number of segments to divide the series into.

    Returns:
        dict: A dictionary удовольствие with feature names as keys (e.g., "seg_1_mean", "seg_1_max")
              and their calculated values. Returns empty dict if n_segments is invalid.
    """
    if not isinstance(n_segments, int) or n_segments <= 0:
        # print("Warning: n_segments must be a positive integer.")
        return {} # Return empty if n_segments is invalid

    features = {}
    # Initialize all potential segment features to NaN.
    # This ensures that if a segment can't be processed (e.g., it's empty),
    # its corresponding features will have NaN values.
    for i in range(n_segments):
        features[f'seg_{i+1}_mean'] = np.nan
        features[f'seg_{i+1}_max'] = np.nan
        features[f'seg_{i+1}_min'] = np.nan

    if len(series) == 0: # If the input series is empty, return NaNs for all segment features.
        return features

    # np.array_split behaves as follows:
    # - If len(series) < n_segments, it will create n_segments arrays,
    #   some of which will be empty. The non-empty ones will contain one element each.
    # - If len(series) >= n_segments, it will distribute the elements among n_segments arrays.
    segments = np.array_split(series, n_segments)

    for i, segment in enumerate(segments):
        if len(segment) > 0: # Only calculate features if the segment is not empty
            features[f'seg_{i+1}_mean'] = np.mean(segment)
            features[f'seg_{i+1}_max'] = np.max(segment)
            features[f'seg_{i+1}_min'] = np.min(segment)
        # If a segment is empty (e.g., len(series)=1, n_segments=3, seg_2 and seg_3 will be empty),
        # its features will remain NaN as initialized.

    return features

# --- Feature Extractor Dispatcher ---
# This dictionary maps feature names (as they might be selected in UI) to functions
AVAILABLE_FEATURES = {
    "mean": calculate_mean,
    "std": calculate_std,
    "max": calculate_max,
    "min": calculate_min,
    "rms": calculate_rms,
    # "segmented_features" is a special case handled separately due to n_segments
}

def extract_features(series, selected_feature_names, n_segments=None):
    """
    Extracts specified features from a series.

    Args:
        series (pd.Series or np.array): The input time series data.
        selected_feature_names (list): A list of strings энергия of features to extract.
                                     Example: ["mean", "std", "segmented_features"]
        n_segments (int, optional): Number of segments for "segmented_features".
                                    Required if "segmented_features" is in selected_feature_names.

    Returns:
        dict: A dictionary where keys are feature names and values are the calculated feature values.
    """
    extracted_values = {}
    for feature_name in selected_feature_names:
        if feature_name in AVAILABLE_FEATURES:
            try:
                extracted_values[feature_name] = AVAILABLE_FEATURES[feature_name](series)
            except Exception as e:
                # print(f"Error calculating basic feature {feature_name}: {e}")
                extracted_values[feature_name] = np.nan
        elif feature_name == "segmented_features":
            if n_segments is not None and isinstance(n_segments, int) and n_segments > 0:
                try:
                    segment_features = calculate_segmented_features(series, n_segments)
                    extracted_values.update(segment_features) # Merge dicts
                except Exception as e:
                    # print(f"Error calculating segmented features: {e}")
                    # Add NaNs for all potential segmented features if error
                    for i in range(n_segments):
                        extracted_values[f'seg_{i+1}_mean'] = np.nan
                        extracted_values[f'seg_{i+1}_max'] = np.nan
                        extracted_values[f'seg_{i+1}_min'] = np.nan
            else:
                # print("Warning: 'segmented_features' selected but n_segments is invalid or not provided.")
                # Potentially add NaN for expected segment features if n_segments was intended but invalid
                pass # Or handle by adding NaNs for a default/expected number of segments
        # else:
            # print(f"Warning: Feature '{feature_name}' not recognized.")

    return extracted_values

if __name__ == '__main__':
    # Example Usage:
    sample_series_short = pd.Series([1, 2, 3, 4, 5])
    sample_series_long = pd.Series(np.random.rand(100))

    print("--- Basic Features (Short Series) ---")
    features_short = extract_features(sample_series_short, ["mean", "max", "min", "std", "rms"])
    print(features_short)

    print("\n--- Segmented Features (Short Series, 3 Segments) ---")
    # Test case where series length < n_segments
    features_segmented_short_3 = extract_features(sample_series_short, ["segmented_features"], n_segments=3)
    print(features_segmented_short_3)

    print("\n--- Segmented Features (Short Series, 7 Segments - more than length) ---")
    features_segmented_short_7 = extract_features(sample_series_short, ["segmented_features"], n_segments=7)
    print(features_segmented_short_7)

    print("\n--- All Features (Long Series, 5 Segments) ---")
    features_all_long = extract_features(sample_series_long, ["mean", "std", "segmented_features"], n_segments=5)
    print(features_all_long)

    print("\n--- Segmented Features (Long Series, 0 Segments - invalid) ---")
    features_segmented_invalid = extract_features(sample_series_long, ["segmented_features"], n_segments=0)
    print(features_segmented_invalid) # Should be empty or handle appropriately

    print("\n--- Segmented Features (Long Series, None Segments - invalid for segmented) ---")
    features_segmented_none = extract_features(sample_series_long, ["segmented_features"], n_segments=None)
    print(features_segmented_none)

    print("\n--- Unknown Feature ---")
    features_unknown = extract_features(sample_series_long, ["unknown_feature", "mean"])
    print(features_unknown)

    # Test with a series of length exactly n_segments
    sample_series_exact = pd.Series([10,20,30])
    print("\n--- Segmented Features (Series length == n_segments) ---")
    features_segmented_exact = extract_features(sample_series_exact, ["segmented_features"], n_segments=3)
    print(features_segmented_exact)

    # Test with a series slightly longer than n_segments
    sample_series_slightly_longer = pd.Series([10,20,30,40])
    print("\n--- Segmented Features (Series slightly longer than n_segments) ---")
    features_segmented_slightly_longer = extract_features(sample_series_slightly_longer, ["segmented_features"], n_segments=3)
    print(features_segmented_slightly_longer)

    # Test with empty series
    empty_series = pd.Series([], dtype=float)
    print("\n--- Basic Features (Empty Series) ---")
    empty_basic = extract_features(empty_series, ["mean", "max"]) # Max will raise error, mean gives nan
    print(empty_basic) # numpy.Max will error on empty, mean will be NaN. Should handle.

    print("\n--- Segmented Features (Empty Series) ---")
    empty_segmented = extract_features(empty_series, ["segmented_features"], n_segments=3)
    print(empty_segmented)

    # Refined test for empty series handling within extract_features
    print("\n--- Refined Basic Features (Empty Series) ---")
    # Modify extract_features to catch exceptions from basic funcs and return NaN
    # For now, this will show existing behavior
    # Mean gives RuntimeWarning and NaN. Max/Min error. Std gives NaN. RMS gives NaN.
    # This indicates that individual feature functions should ideally handle empty inputs or
    # the main `extract_features` should wrap calls in try-except for robustness.
    # The current `extract_features` has try-except for basic features.

    # After adding try-except in extract_features for basic ones:
    # (Assuming the print statements inside extract_features are commented out for cleaner test output)
    # features_short: {'mean': 3.0, 'max': 5, 'min': 1, 'std': 1.4142135623730951, 'rms': 3.3166247903554}
    # features_segmented_short_3: {'seg_1_mean': 1.5, 'seg_1_max': 2, 'seg_1_min': 1, 'seg_2_mean': 3.5, 'seg_2_max': 4, 'seg_2_min': 3, 'seg_3_mean': 5.0, 'seg_3_max': 5, 'seg_3_min': 5}
    # features_segmented_short_7: {'seg_1_mean': 1.0, ..., seg_5_mean: 5.0, seg_6_mean: nan, seg_6_max: nan, seg_6_min: nan, ...}
    # empty_basic: {'mean': nan, 'max': nan} (if max/min are caught and return nan)
    # empty_segmented: {'seg_1_mean': nan, 'seg_1_max': nan, 'seg_1_min': nan, ...}

    # Let's re-run the empty series test after confirming try-except in extract_features
    # (The provided code for extract_features already has try-except for basic features)
    print("\n--- Basic Features (Empty Series - after confirming try-except) ---")
    empty_basic_refined = extract_features(empty_series, ["mean", "max", "min", "std", "rms"])
    print(empty_basic_refined)

    print("\n--- Segmented Features (Empty Series - after confirming try-except) ---")
    empty_segmented_refined = extract_features(empty_series, ["segmented_features"], n_segments=3)
    print(empty_segmented_refined)

    # Example of extracting multiple types of features including segmented
    sample_series = pd.Series(np.arange(1, 21)) # Length 20
    print(f"\n--- Combined Feature Extraction (Series length 20, 4 segments) ---")
    combined_features = extract_features(
        series=sample_series,
        selected_feature_names=["mean", "rms", "segmented_features", "std"],
        n_segments=4
    )
    print(combined_features)
    # Expected: mean, rms, std, and seg_1_mean to seg_4_min (12 features from segments)
    # Total 3 + 12 = 15 features if all are calculated.
    # Check: mean, rms, std are single values. Segmented features has 4*3=12 values.
    # Total 1 + 1 + 1 + 12 = 15 keys in the dict.

    print(f"Number of features from combined: {len(combined_features.keys())}")

    print("\n--- Test with series length 1, n_segments 3 ---")
    series_len1 = pd.Series([100])
    features_len1_seg3 = extract_features(series_len1, ["segmented_features"], n_segments=3)
    print(features_len1_seg3)
    # Expect: seg_1_mean:100, seg_1_max:100, seg_1_min:100, others NaN

    print("\n--- Test with series length 2, n_segments 3 ---")
    series_len2 = pd.Series([100, 200])
    features_len2_seg3 = extract_features(series_len2, ["segmented_features"], n_segments=3)
    print(features_len2_seg3)
    # Expect: seg_1_mean:100, seg_1_max:100, seg_1_min:100,
    #         seg_2_mean:200, seg_2_max:200, seg_2_min:200,
    #         seg_3_mean:nan, seg_3_max:nan, seg_3_min:nan

    print("\n--- Test with series length 5, n_segments 3 (re-check from above) ---")
    # sample_series_short = pd.Series([1, 2, 3, 4, 5])
    features_segmented_short_3_recheck = extract_features(sample_series_short, ["segmented_features"], n_segments=3)
    print(features_segmented_short_3_recheck)
    # seg1 = [1,2], seg2=[3,4], seg3=[5]
    # seg_1_mean = 1.5, seg_1_max = 2, seg_1_min = 1
    # seg_2_mean = 3.5, seg_2_max = 4, seg_2_min = 3
    # seg_3_mean = 5.0, seg_3_max = 5, seg_3_min = 5
    # This is how np.array_split([1,2,3,4,5], 3) works: array([1, 2]), array([3, 4]), array([5])
    # The code's logic for `calculate_segmented_features` seems to align with this.
