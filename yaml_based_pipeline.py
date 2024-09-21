import pandas as pd
import numpy as np
import yaml
import os

# 1. Load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 2. Data Ingestion based on config
def fetch_eurusd_data(config):
    date_rng = pd.date_range(start=config['data_ingestion']['start_date'], end=config['data_ingestion']['end_date'], freq='D')
    data = pd.DataFrame(date_rng, columns=['date'])
    data['price'] = [1.05, 1.06, np.nan, 1.08, 1.07, 1.06, 1.07, 1.08, 1.09, 1.10]  # Simulated data
    data.set_index('date', inplace=True)
    return data

# 3. Data Preprocessing (based on NaN handling in config)
def preprocess_data(data, config):
    nan_handling = config['preprocessing']['nan_handling']
    if nan_handling == 'ffill':
        data['price'].fillna(method='ffill', inplace=True)  # Forward fill
    elif nan_handling == 'bfill':
        data['price'].fillna(method='bfill', inplace=True)  # Backward fill
    return data

# 4. Feature Engineering (Create Moving Average based on config)
def create_features(data, config):
    window_size = config['feature_engineering']['moving_average_window']
    data['MA' + str(window_size)] = data['price'].rolling(window=window_size).mean()
    return data

# 5. Model Training (Use last known value for prediction)
def train_model(data, config):
    if config['model']['type'] == "last_known_value":
        data['predict'] = data['price'].shift(1)  # Simple "last known value" prediction model
    return data

# 6. Model Evaluation (Calculate error based on config)
def evaluate_model(data, config):
    data['error'] = data['price'] - data['predict']  # Error between actual and predicted
    print("Evaluation:\n", data[['price', 'predict', 'error']])
    return data

# 7. Drift Detection (Compare drift between values if enabled in config)
def detect_drift(data, config):
    if config['drift_detection']['enabled']:
        drift_threshold = config['drift_detection']['drift_threshold']
        data['drift'] = data['price'] - data['price'].shift(1)  # Calculate drift as difference between periods
        drift_detected = data['drift'].abs().mean() > drift_threshold  # Check if drift is above threshold
        print("Drift detection over time:\n", data[['price', 'drift']])
        print(f"Drift detected: {drift_detected}")
        return drift_detected
    return False

# 8. Model Deployment (based on config)
def deploy_model(data, config):
    if config['deployment']['save_model']:
        os.makedirs('model', exist_ok=True)
        # Save the model here (for now, just simulating deployment)
        print("Model saved for future use.")
    print("Model deployed.")

# 9. Run Pipeline using the configuration
def run_pipeline(config):
    # 1. Fetch Data
    data = fetch_eurusd_data(config)
    
    # 2. Preprocess Data
    data = preprocess_data(data, config)
    
    # 3. Feature Engineering
    data = create_features(data, config)
    
    # 4. Train Model
    data = train_model(data, config)
    
    # 5. Evaluate Model
    data = evaluate_model(data, config)
    
    # 6. Drift Detection
    detect_drift(data, config)
    
    # 7. Deploy Model
    deploy_model(data, config)

# Main function to run everything
if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)
