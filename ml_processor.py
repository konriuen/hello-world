import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_column='result_numeric'):
    """
    Prepares data for machine learning.
    - Separates features (X) and target (y).
    - Handles missing values in features using mean imputation.
    - Scales numerical features.
    - Splits data into training and testing sets.
    """
    if df.empty or target_column not in df.columns:
        return None, None, None, None, None, "Data is empty or target column missing."

    # Ensure target is numeric (already done by numerize_result, but good check)
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        # This case should ideally not happen if numerize_result was applied
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    X = df.drop(columns=[target_column, 'result', 'sensor_id', 'location_id'], errors='ignore')
    y = df[target_column]

    if X.empty:
        return None, None, None, None, None, "No features available after dropping identifier/target columns."

    # Impute missing values for all feature columns
    # This handles NaNs that might have resulted from feature extraction (e.g., empty segments)
    imputer = SimpleImputer(strategy='mean') # or 'median', 'most_frequent'
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Identify numerical features for scaling (all are numeric after imputation from SimpleImputer)
    numerical_features = X_imputed.select_dtypes(include=np.number).columns

    if len(numerical_features) > 0:
        scaler = StandardScaler()
        X_scaled = X_imputed.copy()
        X_scaled[numerical_features] = scaler.fit_transform(X_imputed[numerical_features])
    else:
        X_scaled = X_imputed # No numerical features to scale

    # Check if target has more than one class, otherwise train_test_split might fail or be meaningless
    if len(y.unique()) < 2:
        return None, None, None, None, None, f"Target variable '{target_column}' has only one class. Cannot train model."

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError as e: # Handles cases where stratification is not possible (e.g. too few samples in a class)
         return None, None, None, None, None, f"Could not split data (e.g. too few samples for stratification): {str(e)}. Try with more data."


    return X_train, X_test, y_train, y_test, X.columns, None # Return original feature names

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, feature_names):
    """
    Trains a specified model and returns evaluation metrics and feature importances.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "LightGBM": lgb.LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=200,      # Increased
            learning_rate=0.05,    # Kept
            num_leaves=25,         # Increased
            min_child_samples=5,   # Decreased
            min_split_gain=0.0,    # Kept
            max_depth=7,           # Added
            verbosity=-1           # Kept
        )
    }

    if model_name not in models:
        return {"error": f"Model {model_name} not supported."}

    model = models[model_name]

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON serialization
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Feature importances/coefficients
        importances = {}
        if hasattr(model, 'feature_importances_'):
            # For Random Forest, LightGBM
            importances_values = model.feature_importances_
            importances = {name: imp for name, imp in zip(feature_names, importances_values)}
        elif hasattr(model, 'coef_'):
            # For Logistic Regression (absolute coefficients as importance)
            importances_values = np.abs(model.coef_[0])
            importances = {name: imp for name, imp in zip(feature_names, importances_values)}

        # Sort importances by value
        sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "roc_auc": roc_auc,
            "feature_importances": sorted_importances,
            "error": None
        }
    except Exception as e:
        return {"model_name": model_name, "error": str(e)}


if __name__ == '__main__':
    # Create a dummy DataFrame for testing
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100) * 10,
        'feature3': np.random.rand(100) - 0.5,
        'seg_1_mean': np.random.rand(100), # Example segmented feature
        'result_numeric': np.random.choice([0, 1], size=100, p=[0.7, 0.3]), # Imbalanced
        'result': ['OK' if x == 0 else 'NG' for x in np.random.choice([0, 1], size=100)],
        'sensor_id': [f's{i}' for i in range(100)],
        'location_id': np.random.choice(['A', 'B'], size=100)
    }
    # Add some NaNs to test imputer
    data['feature1'][0:5] = np.nan

    dummy_df = pd.DataFrame(data)

    print("--- Testing Preprocessing ---")
    X_train, X_test, y_train, y_test, feature_cols, error_msg = preprocess_data(dummy_df.copy()) # Use .copy()
    if error_msg:
        print(f"Preprocessing Error: {error_msg}")
    else:
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train distribution: {y_train.value_counts(normalize=True)}")
        print(f"y_test distribution: {y_test.value_counts(normalize=True)}")
        print(f"Feature columns: {feature_cols}")

        print("\n--- Testing Model Training & Evaluation ---")
        for model_name_test in ["Logistic Regression", "Random Forest", "LightGBM"]:
            print(f"\n-- {model_name_test} --")
            results = train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name_test, feature_cols)
            if results.get("error"):
                print(f"Error: {results['error']}")
            else:
                print(f"Accuracy: {results['accuracy']:.4f}")
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall: {results['recall']:.4f}")
                print(f"F1 Score: {results['f1_score']:.4f}")
                print(f"ROC AUC: {results['roc_auc']:.4f}")
                print(f"Confusion Matrix: {results['confusion_matrix']}")
                print(f"Top 3 Feature Importances: {dict(list(results['feature_importances'].items())[:3])}")

    print("\n--- Test with insufficient data for stratification ---")
    data_few_samples = {
        'feature1': [1,2,3,4],
        'result_numeric': [0,0,1,1], # Only 2 per class
        'result': ['OK','OK','NG','NG'],
        'sensor_id': ['s1','s2','s3','s4'],
        'location_id': ['A','A','A','A']
    }
    dummy_df_few = pd.DataFrame(data_few_samples)
    _, _, _, _, _, error_msg_few = preprocess_data(dummy_df_few.copy())
    if error_msg_few:
        print(f"Preprocessing Error (few samples): {error_msg_few}")

    print("\n--- Test with only one class in target ---")
    data_one_class = {
        'feature1': [1,2,3,4,5,6],
        'result_numeric': [0,0,0,0,0,0],
        'result': ['OK','OK','OK','OK','OK','OK'],
        'sensor_id': ['s1','s2','s3','s4','s5','s6'],
        'location_id': ['A','A','A','A','A','A']
    }
    dummy_df_one_class = pd.DataFrame(data_one_class)
    _, _, _, _, _, error_msg_one_class = preprocess_data(dummy_df_one_class.copy())
    if error_msg_one_class:
        print(f"Preprocessing Error (one class): {error_msg_one_class}")

    print("\n--- Test with empty dataframe ---")
    empty_df = pd.DataFrame()
    _, _, _, _, _, error_msg_empty = preprocess_data(empty_df.copy())
    if error_msg_empty:
        print(f"Preprocessing Error (empty df): {error_msg_empty}")

    print("\n--- Test with no features ---")
    no_features_df = pd.DataFrame({
        'result_numeric': [0,0,1,1],
        'result': ['OK','OK','NG','NG'],
        'sensor_id': ['s1','s2','s3','s4'],
        'location_id': ['A','A','A','A']
    })
    _, _, _, _, _, error_msg_no_features = preprocess_data(no_features_df.copy())
    if error_msg_no_features:
        print(f"Preprocessing Error (no features): {error_msg_no_features}")
