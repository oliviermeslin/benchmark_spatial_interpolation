import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph

# --- START: Definitions from utils.py ---
epsilon = 1e-10

def local_moranI(y, w):
  y = y.reshape(-1)
  n = len(y)
  n_1 = n - 1
  z = y - y.mean()
  sy = max(y.std(), epsilon)
  z /= sy
  den = max((z * z).sum(), epsilon)
  zl = w * z
  mi = n_1 * z * zl / den
  scaler = MinMaxScaler()
  return scaler.fit_transform(mi.reshape(-1,1)).reshape(-1)


def row_standardize(w):
  new_data = []
  for i in range(0, w.shape[0]):
    wijs = w.getrow(i).data
    row_sum = sum(wijs)
    if row_sum == 0.0:
        print(("WARNING: ", i, " is an island (no neighbors)"))
    new_data.append([wij / row_sum for wij in wijs])
  w.data = np.array(new_data).flatten()
  return w

def calc_w(locations, k, weighted):
    mode = 'distance' if weighted else 'connectivity'
    w= kneighbors_graph(locations, n_neighbors=k, mode=mode, metric='haversine', include_self=False)
    # fill in epsilon value where data values is 0
    w.data = w.data + epsilon
    w.data = w.data**(-1)
    return row_standardize(w)

def smape(y_true, y_pred):
    den = np.max([(np.abs(y_pred) + np.abs(y_true)),np.array([epsilon]*len(y_pred))], axis=0)
    return np.mean( 
            np.abs(y_pred - y_true) / 
            (den/2))
# --- END: Definitions from utils.py ---

# --- START: Definitions from MIBooster.py ---
import lightgbm
import xgboost

def pseudohuber_moranI_loss(y_pred, y_true, w):
  i_pred = local_moranI(y_pred, w)
  i_true = local_moranI(y_true, w)
  d = (i_true-i_pred)
  delta = 1  
  scale = 1 + (d / delta) ** 2
  scale_sqrt = np.sqrt(scale)
  grad = d / scale_sqrt 
  hess = (1 / scale) / scale_sqrt
  return grad, hess

def mse_moranI_loss(y_pred, y_true, w):
  i_pred = local_moranI(y_pred, w)
  i_true = local_moranI(y_true, w)
  grad = 2*(i_true-i_pred)
  hess = np.repeat(2,i_true.shape[0])
  return grad, hess

def mse_loss(y_pred, y_true):
  grad = 2*(y_true-y_pred)
  hess = np.repeat(2,y_true.shape[0])
  return grad, hess  

def weighted_loss1(y_pred, y_true, w, l=0.01):
  grad_mi, hess_mi = pseudohuber_moranI_loss(y_pred, y_true, w)
  grad_mse, hess_mse = mse_loss(y_pred, y_true)
  return grad_mse + l*grad_mi, hess_mse + l*hess_mi

def weighted_loss2(y_pred, y_true, w, l=0.01):
  grad_mi, hess_mi = mse_moranI_loss(y_pred, y_true, w)
  grad_mse, hess_mse = mse_loss(y_pred, y_true)
  return grad_mse + l*grad_mi, hess_mse + l*hess_mi


class MILGBooster(lightgbm.LGBMRegressor):
    def __init__(self, k=5, weighted=True, lamb= 0.01, loss_type='mse',
        boosting_type='gbdt', num_leaves=31, max_depth=-1, 
        learning_rate=0.1, n_estimators=100, 
        subsample_for_bin=200000, objective=None, class_weight=None, 
        min_split_gain=0.0, min_child_weight=0.001, 
        min_child_samples=20, subsample=1.0, subsample_freq=0, 
        colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, 
        random_state=None, n_jobs=None, importance_type='split', 
        **kwargs):
        super().__init__(boosting_type=boosting_type, num_leaves=num_leaves,
        max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
        subsample_for_bin=subsample_for_bin, objective=objective, class_weight=class_weight,
        min_split_gain=min_split_gain, min_child_weight=min_child_weight,
        min_child_samples=min_child_samples, subsample=subsample, subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        random_state=random_state, n_jobs=n_jobs, importance_type=importance_type,
         **kwargs)
        self.k = k
        self.lamb = lamb
        self.loss_type=loss_type
        self.w = None
        self.weighted = weighted
        

    def weighted_loss(self, y_pred, y_true):
        if self.loss_type == 'mse':
            return weighted_loss2(y_pred, y_true, w=self.w, l=self.lamb)
        else:
            return weighted_loss1(y_pred, y_true, w=self.w, l=self.lamb)

    def set_objective(self):
        self.set_params(objective=self.weighted_loss)


    def fit(self, X, y, locs=None):
        locations = locs if locs is not None else np.deg2rad(X[:,[-1,-2]])
        self.w = calc_w(locations, k=self.k, weighted=self.weighted) if self.w is None else self.w
        self.set_objective()
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)

class MIXGBooster(xgboost.XGBRegressor):
    def __init__(self, k=5, weighted=True, lamb= 0.01, loss_type='mse',
        **kwargs):
        super().__init__( **kwargs)
        self.k = k
        self.lamb = lamb
        self.loss_type=loss_type
        self.w = None
        self.weighted = weighted
        

    def weighted_loss(self, y_pred, y_true):
        if self.loss_type == 'mse':
            return weighted_loss2(y_pred, y_true, w=self.w, l=self.lamb)
        else:
            return weighted_loss1(y_pred, y_true, w=self.w, l=self.lamb)

    def set_objective(self):
        self.set_params(objective=self.weighted_loss)

    def fit(self, X, y, locs=None):
        locations = locs if locs is not None else np.deg2rad(X[:,[-1,-2]])
        self.w = calc_w(locations, k=self.k, weighted=self.weighted) if self.w is None else self.w
        self.set_objective()
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)
# --- END: Definitions from MIBooster.py ---

# --- START: Synthetic Data and Main Logic (user-defined functions) ---
def generate_synthetic_data(n_samples: int = 2500) -> pd.DataFrame:
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
    # Use a grid-like pattern to ensure clear spatial structure and autocorrelation
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
    df['feature_1'] = np.random.rand(n_samples) 

    # 3. Generate the target variable 'altitude' (z)
    # A smooth function of coordinates (spatial autocorrelation) + feature_1 + noise.
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
                 loss_type: str = 'pseudohuber') -> MIXGBooster:
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
        model = MIXGBooster(
            k=k, 
            lamb=lamb, 
            loss_type=loss_type, 
            n_estimators=100, 
            learning_rate=0.1, 
            verbosity=0
        )
    elif booster_type == 'LGB':
        model = MILGBooster(
            k=k, 
            lamb=lamb, 
            loss_type=loss_type, 
            n_estimators=100, 
            learning_rate=0.1, 
            n_jobs=-1,
            verbose=-1
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
    smap_error = smape(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'SMAPE': smap_error
    }
    
    print(f"\n--- {dataset_name} Metrics ---")
    print(f"RMSE:  {metrics['RMSE']:.6f}")
    print(f"SMAPE: {metrics['SMAPE']:.6f}")
    
    return metrics

def train_baseline_gbt(X_train: np.ndarray, y_train: np.ndarray, booster_type: str = 'XGB') -> xgboost.XGBRegressor:
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
        model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.1, verbosity=0)
    elif booster_type == 'LGB':
        model = lightgbm.LGBMRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, verbose=-1)
    else:
        raise ValueError("booster_type must be 'XGB' or 'LGB'")
    
    print(f"Training {booster_type} Baseline model...")
    model.fit(X_train, y_train)
    return model

def run_demo():
    """
    Main function to run the synthetic spatial regression demo.
    """
    print("--- 1. Generating Synthetic Spatial Data ---")
    synthetic_df = generate_synthetic_data(n_samples=2500)
    print(f"Generated {len(synthetic_df)} samples.")
    
    # Save the synthetic data to a CSV file for inspection
    synthetic_df.to_csv("synthetic_spatial_data.csv", index=False)
    
    print("\n--- 2. Preparing Data for Training ---")
    X_train, y_train, locs_train, X_val, y_val, locs_val, X_test, y_test, locs_test = prepare_data(
        synthetic_df, target_col='altitude'
    )
    print(f"Train samples: {len(y_train)}, Validation samples: {len(y_val)}, Test samples: {len(y_test)}")
    
    # --- 3. Train and Evaluate MI-GBT Model (XGBoost implementation) ---
    mi_gbt_model = train_mi_gbt(
        X_train, y_train, locs_train, 
        k=35, 
        lamb=0.1, 
        booster_type='XGB',
        loss_type='pseudohuber'
    )
    
    # --- 4. Train and Evaluate Baseline Model (XGBoost implementation) ---
    baseline_model = train_baseline_gbt(X_train, y_train, booster_type='XGB')

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

# Run the demonstration
run_demo()
# --- END: Execution ---