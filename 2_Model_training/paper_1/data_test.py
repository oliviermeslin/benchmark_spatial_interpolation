import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import smape
from MIBooster import MIXGBooster, MILGBooster

# The provided code
# --- Paste in the relevant functions from the uploaded files to make the script self-contained ---

# Since the prompt asks for a single file with *many methods*, 
# I will define the methods for data generation, preparation, training, and evaluation.

def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generates a synthetic spatial dataset with a target variable (altitude) 
    that exhibits spatial autocorrelation.
    
    The altitude (z) is a function of latitude (y), longitude (x), and 
    a non-spatial feature (feature_1), with added random noise.
    
    Args:
        n_samples: The total number of data points to generate.
        
    Returns:
        A pandas DataFrame containing the synthetic data.
    """
    # 1. Generate spatial coordinates (Latitude 'y' and Longitude 'x')
    # Use a grid-like pattern to ensure clear spatial structure
    side = int(np.sqrt(n_samples))
    n_samples = side * side # Adjust to a perfect square
    
    # Coordinates in a defined area (e.g., a local region)
    y = np.linspace(40.0, 42.0, side) # Latitude
    x = np.linspace(-75.0, -73.0, side) # Longitude
    
    X_mesh, Y_mesh = np.meshgrid(x, y)
    
    df = pd.DataFrame({
        'y': Y_mesh.flatten(),
        'x': X_mesh.flatten()
    })

    # 2. Generate a non-spatial feature
    # Uniformly distributed random feature
    df['feature_1'] = np.random.rand(n_samples) 

    # 3. Generate the target variable 'altitude' (z) with spatial autocorrelation
    # A function of coordinates that is smooth, plus the non-spatial feature
    # and some noise.
    altitude = (
        5 * np.sin(np.deg2rad(df['y'])) + 
        3 * np.cos(np.deg2rad(df['x'])) + 
        2 * df['feature_1'] + 
        np.random.normal(0, 0.1, n_samples) # Add Gaussian noise
    )

    df['altitude'] = altitude
    
    # Scale the target to a [0, 1] range as done in the provided load_spatialdata.py
    scaler = MinMaxScaler()
    df['altitude'] = scaler.fit_transform(df['altitude'].values.reshape(-1, 1))

    return df

def prepare_data(df: pd.DataFrame, target_col: str = 'altitude', test_size: float = 0.3, random_state: int = 42) -> tuple:
    """
    Splits the synthetic dataset into training, validation, and test sets, 
    and separates features (X), target (y), and locations (locs).
    
    Args:
        df: The input DataFrame.
        target_col: The name of the target variable column.
        test_size: The proportion of the data to allocate to the combined 
                   validation and test sets (e.g., 0.3 for 30%).
        random_state: Random state for reproducibility.
        
    Returns:
        A tuple (X_train, y_train, locs_train, X_val, y_val, locs_val, X_test, y_test, locs_test)
    """
    
    # Define feature columns - exclude 'y', 'x', and the target
    feature_cols = [col for col in df.columns if col not in ['y', 'x', target_col]]
    
    # Split into train and remaining (val + test)
    data_train, data_temp = train_test_split(df, test_size=test_size, shuffle=True, random_state=random_state)
    
    # Split remaining into validation and test (half each)
    data_val, data_test = train_test_split(data_temp, test_size=0.5, shuffle=True, random_state=random_state)
    
    # Prepare X, y, and locs for each set
    X_train = data_train[feature_cols].values
    y_train = data_train[target_col].values
    # Locations must be converted to radians as per MIBooster.py fit method's default
    locs_train = np.deg2rad(data_train[['y', 'x']].values)
    
    X_val = data_val[feature_cols].values
    y_val = data_val[target_col].values
    locs_val = np.deg2rad(data_val[['y', 'x']].values)
    
    X_test = data_test[feature_cols].values
    y_test = data_test[target_col].values
    locs_test = np.deg2rad(data_test[['y', 'x']].values)
    
    # Scale non-spatial features as done in the provided load_spatialdata.py
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, locs_train, X_val, y_val, locs_val, X_test, y_test, locs_test


def train_mi_gbt(X_train: np.ndarray, y_train: np.ndarray, locs_train: np.ndarray, 
                 k: int = 15, lamb: float = 0.1, booster_type: str = 'XGB', 
                 loss_type: str = 'pseudohuber') -> [MIXGBooster, MILGBooster]:
    """
    Trains the MI-GBT model using either XGBoost (MIXGBooster) or LightGBM (MILGBooster).
    
    Args:
        X_train: Training features.
        y_train: Training target.
        locs_train: Training locations (in radians).
        k: Number of neighbors for the spatial weight matrix W.
        lamb: Regularization parameter for the spatial loss component.
        booster_type: 'XGB' or 'LGB'.
        loss_type: 'mse' or 'pseudohuber' (for Moran's I loss).
        
    Returns:
        The trained MI-GBT model.
    """
    
    if booster_type == 'XGB':
        # MIXGBooster implementation
        model = MIXGBooster(
            k=k, 
            lamb=lamb, 
            loss_type=loss_type, 
            n_estimators=100, 
            learning_rate=0.1, 
            verbosity=0
        )
    elif booster_type == 'LGB':
        # MILGBooster implementation
        model = MILGBooster(
            k=k, 
            lamb=lamb, 
            loss_type=loss_type, 
            n_estimators=100, 
            learning_rate=0.1, 
            n_jobs=-1,
            verbose=-1 # Suppress lightgbm output
        )
    else:
        raise ValueError("booster_type must be 'XGB' or 'LGB'")

    print(f"Training {booster_type} MI-GBT with k={k}, lambda={lamb}, loss_type='{loss_type}'...")
    model.fit(X_train, y_train, locs=locs_train)
    
    return model

def evaluate_model(model, X: np.ndarray, y_true: np.ndarray, dataset_name: str) -> dict:
    """
    Evaluates the trained model on a given dataset.
    
    Args:
        model: The trained MI-GBT or baseline model.
        X: Features of the evaluation set.
        y_true: True target values.
        dataset_name: Name of the dataset (e.g., 'Validation').
        
    Returns:
        A dictionary of evaluation metrics.
    """
    y_pred = model.predict(X)
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Calculate SMAPE (Symmetric Mean Absolute Percentage Error) from utils.py
    # Note: SMAPE is typically used for time series, but is available in utils.py
    # and is a scale-independent metric.
    smap_error = smape(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'SMAPE': smap_error
    }
    
    print(f"\n--- {dataset_name} Metrics ---")
    print(f"RMSE:  {metrics['RMSE']:.6f}")
    print(f"SMAPE: {metrics['SMAPE']:.6f}")
    
    return metrics

def train_baseline_gbt(X_train: np.ndarray, y_train: np.ndarray, booster_type: str = 'XGB') -> [pd.DataFrame]:
    """
    Trains a baseline Gradient Boosted Tree model (XGBRegressor or LGBMRegressor).
    
    Args:
        X_train: Training features.
        y_train: Training target.
        booster_type: 'XGB' or 'LGB'.
        
    Returns:
        The trained baseline model.
    """
    if booster_type == 'XGB':
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, verbosity=0)
    elif booster_type == 'LGB':
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, verbose=-1)
    else:
        raise ValueError("booster_type must be 'XGB' or 'LGB'")
    
    print(f"Training {booster_type} Baseline model...")
    model.fit(X_train, y_train)
    return model

def main():
    """
    Main function to run the synthetic spatial regression demo.
    """
    print("--- 1. Generating Synthetic Spatial Data ---")
    # Generate the dataset (Target: 'altitude', Features: 'feature_1', Locations: 'y', 'x')
    synthetic_df = generate_synthetic_data(n_samples=2500)
    print(f"Generated {len(synthetic_df)} samples.")
    print(synthetic_df.head())
    
    # Save the synthetic data to a CSV file for inspection
    synthetic_df.to_csv("synthetic_spatial_data.csv", index=False)
    
    print("\n--- 2. Preparing Data for Training ---")
    # Prepare the splits
    X_train, y_train, locs_train, X_val, y_val, locs_val, X_test, y_test, locs_test = prepare_data(
        synthetic_df, target_col='altitude'
    )
    print(f"Train samples: {len(y_train)}, Validation samples: {len(y_val)}, Test samples: {len(y_test)}")
    
    # --- 3. Train and Evaluate MI-GBT Model (XGBoost implementation) ---
    mi_gbt_model = train_mi_gbt(
        X_train, y_train, locs_train, 
        k=35, # Number of neighbors, similar to example.ipynb
        lamb=0.1, # Spatial regularization strength
        booster_type='XGB',
        loss_type='pseudohuber'
    )
    
    # Evaluate MI-GBT on the Validation Set
    mi_gbt_val_metrics = evaluate_model(mi_gbt_model, X_val, y_val, "MI-GBT Validation")

    # --- 4. Train and Evaluate Baseline Model (XGBoost implementation) ---
    baseline_model = train_baseline_gbt(X_train, y_train, booster_type='XGB')

    # Evaluate Baseline on the Validation Set
    baseline_val_metrics = evaluate_model(baseline_model, X_val, y_val, "Baseline Validation")
    
    # --- 5. Compare Results on Test Set ---
    print("\n--- 5. Final Comparison on Test Set ---")
    
    # Evaluate MI-GBT on the Test Set
    mi_gbt_test_metrics = evaluate_model(mi_gbt_model, X_test, y_test, "MI-GBT Test")
    
    # Evaluate Baseline on the Test Set
    baseline_test_metrics = evaluate_model(baseline_model, X_test, y_test, "Baseline Test")
    
    # Final Summary
    print("\n--- Summary of Test Results (Lower is Better) ---")
    print(f"{'Metric':<8} | {'MI-GBT':<8} | {'Baseline':<8}")
    print("-" * 30)
    print(f"{'RMSE':<8} | {mi_gbt_test_metrics['RMSE']:.4f}   | {baseline_test_metrics['RMSE']:.4f}   ")
    print(f"{'SMAPE':<8} | {mi_gbt_test_metrics['SMAPE']:.4f}   | {baseline_test_metrics['SMAPE']:.4f}   ")

if __name__ == '__main__':
    # Due to the constraints of the environment (not being able to run main), 
    # I will call the main function here to execute the full workflow and demonstrate
    # the process and the results.
    main()

# --- End of Script Content for user's file ---