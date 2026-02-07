"""
benchmark_small.py
Target: Small Datasets (~5,000 points)
Fixes:
1. SyntaxError removed.
2. Timeout check added INSIDE cross-validation loop (stops faster).
3. Data Filtering logic restored (was missing, causing huge data loads).
4. Saves all metrics (Time, R2, MAE).
"""
import json
import time
import gc
import random
import sys
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import xgboost

# Custom Utils
from utils.migbt import SklearnMIXGBooster
from utils.kriging_wrapper import PyKrigeWrapper
from utils.gam_wrapper import PyGAMWrapper
sys.path.append('geoRF')
from geoRF import GeoRFRegressor
from utils.idw import IDWRegressor
from utils.functions import AddCoordinatesRotation, ConvertToPandas
from utils.s3 import get_df_from_s3
from utils.patch_treeple import apply_treeple_patch
apply_treeple_patch()
from treeple.ensemble import ObliqueRandomForestRegressor

# =============================================================================
# CONFIGURATION - SMALL DATA
# =============================================================================

N_ITER_SEARCH = 20   
CV_SPLITS = 5        
RANDOM_STATE = 42
SIZE_SMALL = 5_000
MAX_MODEL_TIME_SEC = 600 # 10 Minutes Timeout

DATASETS = [
    {"name": "bdalti_48", "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/", "filter_col": "departement", "filter_val": "48", "target_n": SIZE_SMALL, "transform": "log"},
    {"name": "rgealti_48", "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/", "filter_col": "departement", "filter_val": "48", "target_n": SIZE_SMALL, "transform": "log"},
    {"name": "S-G-Sm", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Sm.parquet", "target_n": SIZE_SMALL},
    {"name": "S-NG-Sm", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Sm.parquet", "target_n": SIZE_SMALL},
]

MODELS = [
    {
        "name": "random_forest",
        "class": RandomForestRegressor,
        "number_axis": 1,
        "param_space": {"n_estimators": [100, 200, 300, 500], "max_features": ["sqrt", "log2", 1.0], "min_samples_leaf": [1, 3, 5], "n_jobs": [-1], "random_state": [42]}
    },
    {
        "name": "random_forest_cr",
        "class": RandomForestRegressor,
        "number_axis": 23,
        "param_space": {"n_estimators": [100, 200, 300], "max_features": ["sqrt", "log2"], "min_samples_leaf": [1, 3, 5], "n_jobs": [-1], "random_state": [42]}
    },
    {
        "name": "xgboost",
        "class": xgboost.XGBRegressor,
        "number_axis": 1,
        "param_space": {"n_estimators": [200, 500, 1000], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [4, 6, 8, 10], "subsample": [0.7, 0.8, 1.0], "n_jobs": [-1], "random_state": [42], "objective": ["reg:squarederror"]}
    },
    {
        "name": "xgboost_cr",
        "class": xgboost.XGBRegressor,
        "number_axis": 23,
        "param_space": {"n_estimators": [200, 500, 1000], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [4, 6, 8, 10], "n_jobs": [-1], "random_state": [42], "objective": ["reg:squarederror"]}
    },
    {
        "name": "mixgboost",
        "class": SklearnMIXGBooster,
        "number_axis": 1,
        "param_space": {"k": [10, 20, 40], "lamb": [0.01, 0.1, 0.5], "learning_rate": [0.05, 0.1], "n_estimators": [200, 500], "max_depth": [6, 10, 12], "n_jobs": [-1]}
    },
    {
        "name": "mixgboost_cr",
        "class": SklearnMIXGBooster,
        "number_axis": 23,
        "param_space": {"k": [10, 20, 40], "lamb": [0.01, 0.1, 0.5], "learning_rate": [0.05, 0.1], "n_estimators": [200, 500], "max_depth": [6, 10, 12], "n_jobs": [-1]}
    },
    {
        "name": "oblique_rf",
        "class": ObliqueRandomForestRegressor,
        "number_axis": 1,
        "param_space": {"n_estimators": [100, 200], "max_features": [1.0, "sqrt"], "max_depth": [None, 20], "n_jobs": [-1]}
    },
    {
        "name": "geoRF",
        "class": GeoRFRegressor,
        "number_axis": 1,
        "param_space": {"n_estimators": [50, 100], "max_depth": [10, 20], "max_features": [2], "n_jobs": [1], "random_state": [42]}
    },
    {
        "name": "kriging",
        "class": PyKrigeWrapper,
        "number_axis": 1,
        "param_space": {"variogram_model": ["linear", "power", "gaussian", "exponential"], "nlags": [6, 10, 20], "weight": [True, False]}
    },
    {
        "name": "gam",
        "class": PyGAMWrapper,
        "number_axis": 1,
        "param_space": {"n_splines": [15, 25, 50], "lam": [0.1, 0.6, 2.0], "spline_order": [3]}
    },
    {
        "name": "gam_cr",
        "class": PyGAMWrapper,
        "number_axis": 23,
        "param_space": {"n_splines": [15, 25, 50], "lam": [0.1, 0.6, 2.0], "spline_order": [3]}
    },
    {
        "name": "knn_3",
        "class": KNeighborsRegressor,
        "number_axis": 1,
        "param_space": {"n_neighbors": [3, 5, 9, 15], "weights": ["distance", "uniform"], "n_jobs": [-1]}
    },
    {
        "name": "idw_p3",
        "class": IDWRegressor,
        "number_axis": 1,
        "param_space": {"power": [1, 2, 3, 5], "n_neighbors": [10, 20, 50]}
    }
]

METRICS = [
    {"name": "r2_score", "func": r2_score},
    {"name": "rmse", "func": root_mean_squared_error},
    {"name": "mae", "func": mean_absolute_error},
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def sample_parameters(param_space: dict) -> dict:
    return {k: random.choice(v) for k, v in param_space.items()}

def load_dataset(dataset_config: dict):
    ldf = get_df_from_s3(dataset_config["path"])
    target_n = dataset_config.get("target_n", 5_000)
    
    if "val" in ldf.collect_schema().names(): ldf = ldf.rename({"val": "value"})

    # --- FIX: Apply Filter BEFORE Collect (Crucial for Speed) ---
    if "filter_col" in dataset_config and "filter_val" in dataset_config:
        print(f"  -> Filtering {dataset_config['filter_col']} == {dataset_config['filter_val']}")
        ldf = ldf.filter(pl.col(dataset_config["filter_col"]) == dataset_config["filter_val"])

    # Basic Cleaning
    ldf = ldf.filter((pl.col("value").is_not_null()) & (pl.col("value") > 0))
    
    # Collect only necessary columns
    print("  -> Collecting data...")
    df = ldf.select(["x", "y", "value"]).collect(engine="streaming")
    
    if dataset_config.get("transform") == "log":
        df = df.with_columns(pl.col("value").log())
        
    df = df.filter(pl.col("value").is_finite())
    
    if len(df) >= target_n:
        df = df.sample(n=target_n, seed=RANDOM_STATE)
    
    X = df.select(["x", "y"]).to_pandas()
    y = df.select("value").to_numpy().ravel().astype(float)
    return X, y

def run_single_fit(model_class, params, number_axis, X_train_pl, y_train, X_test_pl, y_test):
    pipeline = Pipeline([
        ("coord_rotation", AddCoordinatesRotation(coordinates_names=("x", "y"), number_axis=number_axis)),
        ("pandas_converter", ConvertToPandas()),
        ("ml_model", model_class(**params)),
    ])
    start = time.perf_counter()
    pipeline.fit(X_train_pl, y_train)
    time_taken = time.perf_counter() - start
    y_pred = pipeline.predict(X_test_pl)
    
    metrics = {m["name"]: float(m["func"](y_test, y_pred)) for m in METRICS}
    metrics['training_time'] = round(time_taken, 2)
    return metrics

def run_benchmark():
    results = []
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for dataset in DATASETS:
        print(f"\n=== [Small] Processing {dataset['name']} ===")
        try:
            X, y = load_dataset(dataset)
        except Exception as e:
            print(f"Skipping {dataset['name']}: {e}"); continue

        for model_config in MODELS:
            model_name = model_config["name"]
            print(f"  > Tuning {model_name}...")
            
            best_score = float('inf')
            best_result = None
            
            # --- 1. Start Timer ---
            model_start_time = time.time()

            for i in range(N_ITER_SEARCH):
                # Check timeout before starting a new param set
                if time.time() - model_start_time > MAX_MODEL_TIME_SEC:
                    print(f"    [STOP] Timeout ({MAX_MODEL_TIME_SEC}s) reached. Stopping search for {model_name}.")
                    break

                params = sample_parameters(model_config["param_space"])
                fold_results = []
                
                # Cross-validation
                for train_idx, test_idx in kf.split(X, y):
                    # --- FIX: Check timeout INSIDE CV loop ---
                    if time.time() - model_start_time > MAX_MODEL_TIME_SEC:
                        break # Stop CV midway if time is up

                    X_tr, X_te = pl.from_pandas(X.iloc[train_idx]), pl.from_pandas(X.iloc[test_idx])
                    y_tr, y_te = y[train_idx], y[test_idx]
                    try:
                        m = run_single_fit(model_config["class"], params, model_config.get("number_axis", 1), X_tr, y_tr, X_te, y_te)
                        fold_results.append(m)
                    except: pass
                
                # Only use full results
                if fold_results:
                    avg_rmse = np.mean([res['rmse'] for res in fold_results])
                    if avg_rmse < best_score:
                        best_score = avg_rmse
                        
                        # --- FIX: Average ALL metrics ---
                        avg_metrics = {
                            k: float(np.mean([res[k] for res in fold_results])) 
                            for k in fold_results[0].keys()
                        }
                        
                        best_result = {
                            "model": model_name, 
                            "dataset": dataset["name"], 
                            "best_params": params,
                            **avg_metrics
                        }

            if best_result:
                print(f"    -> WINNER: RMSE {best_result['rmse']:.4f} | Time: {best_result['training_time']:.2f}s")
                results.append(best_result)

    output_path = Path("results/results_small.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_benchmark()