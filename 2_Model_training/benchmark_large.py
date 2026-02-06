"""
benchmark_large.py - CORRECTED & UPDATED
Target: Large Datasets (~100,000 points) & Noisy Datasets
Updates: Uses the 3 new variable-noise synthetic datasets (N1, N2, N3).
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
from utils.idw import IDWRegressor
import xgboost

from utils.migbt import SklearnMIXGBooster
from utils.functions import AddCoordinatesRotation, ConvertToPandas
from utils.s3 import get_df_from_s3
from utils.patch_treeple import apply_treeple_patch
apply_treeple_patch()
from treeple.ensemble import ObliqueRandomForestRegressor

# =============================================================================
# CONFIGURATION
# =============================================================================

N_ITER_SEARCH = 5
CV_SPLITS = 3
RANDOM_STATE = 42
SIZE_LARGE = 100_000
SIZE_VERY_LARGE = 1_000_000
SIZE_EXTREM_LARGE = 10_000_000

DATASETS = [
    # --- Real World Datasets ---
    {"name": "rgealti", "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/", "target_n": SIZE_LARGE, "transform": "log"},
    {"name": "bdalti", "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/", "target_n": SIZE_LARGE, "transform": "log"},
    {"name": "cal_housing", "path": "s3://projet-benchmark-spatial-interpolation/data/real/HOUSING/california_housing.parquet", "target_n": 25000, "noisy": True},

    # --- Clean Synthetic Datasets  ---
    {"name": "S-G-Lg", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Lg.parquet", "target_n": SIZE_LARGE},
    {"name": "S-NG-Lg", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg.parquet", "target_n": SIZE_LARGE},
    {"name": "S-NG-VLg", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg.parquet", "target_n": SIZE_VERY_LARGE},
    {"name": "S-NG-ELg", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg.parquet", "target_n": SIZE_EXTREM_LARGE},

    # --- New Synthetic Datasets (Large, No Grid, Increasing Noise) ---
    # N1 = Std 0.5 (Low Noise)
    {"name": "S-NG-Lg-N1", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg-N1.parquet", "target_n": SIZE_LARGE, "noisy": True},
    # N2 = Std 1.0 (Medium Noise)
    {"name": "S-NG-Lg-N2", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg-N2.parquet", "target_n": SIZE_LARGE, "noisy": True},
    # N3 = Std 2.0 (High Noise)
    {"name": "S-NG-Lg-N3", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg-N3.parquet", "target_n": SIZE_LARGE, "noisy": True},
]

MODELS = [
    {
        "name": "random_forest",
        "class": RandomForestRegressor,
        "number_axis": 1,
        "param_space": {"n_estimators": [150, 250], "max_features": ["sqrt"], "min_samples_leaf": [5, 10], "n_jobs": [-1], "random_state": [42]}
    },
    {
        "name": "random_forest_cr",
        "class": RandomForestRegressor,
        "number_axis": 23,
        "param_space": {"n_estimators": [150, 250], "max_features": ["sqrt"], "min_samples_leaf": [5, 10], "n_jobs": [-1], "random_state": [42]}
    },
    {
        "name": "xgboost",
        "class": xgboost.XGBRegressor,
        "number_axis": 1,
        "param_space": {"n_estimators": [300, 500], "learning_rate": [0.05, 0.1], "max_depth": [6, 8], "n_jobs": [-1], "random_state": [42], "objective": ["reg:squarederror"]}
    },
    {
        "name": "xgboost_cr",
        "class": xgboost.XGBRegressor,
        "number_axis": 23,
        "param_space": {"n_estimators": [300, 500], "learning_rate": [0.05, 0.1], "max_depth": [6, 8], "n_jobs": [-1], "random_state": [42], "objective": ["reg:squarederror"]}
    },
    {
        "name": "mixgboost",
        "class": SklearnMIXGBooster,
        "number_axis": 1,
        "param_space": {"k": [20], "lamb": [0.05], "learning_rate": [0.1], "n_estimators": [300], "max_depth": [8], "n_jobs": [-1]}
    },
    {
        "name": "mixgboost_cr",
        "class": SklearnMIXGBooster,
        "number_axis": 23,
        "param_space": {"k": [20], "lamb": [0.05], "learning_rate": [0.1], "n_estimators": [300], "max_depth": [8], "n_jobs": [-1]}
    },
    {
        "name": "oblique_rf",
        "class": ObliqueRandomForestRegressor,
        "number_axis": 1,
        "param_space": {"n_estimators": [100], "max_features": [1.0], "max_depth": [15], "n_jobs": [-1]}
    },
    {
        "name": "knn_3",
        "class": KNeighborsRegressor,
        "number_axis": 1,
        "param_space": {"n_neighbors": [5, 15], "weights": ["distance"], "n_jobs": [-1]}
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
    target_n = dataset_config.get("target_n", 100_000)
    
    # Normalize column names
    if "val" in ldf.collect_schema().names(): 
        ldf = ldf.rename({"val": "value"})
    
    # Special handling for huge RGEALTI dataset to avoid memory spikes
    if "rgealti" in dataset_config["name"]: 
        ldf = ldf.slice(0, 1_000_000)

    # Basic cleaning
    ldf = ldf.filter((pl.col("value").is_not_null()) & (pl.col("value") > 0))
    df = ldf.select(["x", "y", "value"]).collect(engine="streaming")
    
    # Transformations
    if dataset_config.get("transform") == "log":
        df = df.with_columns(pl.col("value").log())
        
    df = df.filter(pl.col("value").is_finite())
    
    # Sampling
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
        print(f"\n=== [Large/Noisy] Processing {dataset['name']} ===")
        try:
            X, y = load_dataset(dataset)
        except Exception as e:
            print(f"Skipping {dataset['name']}: {e}"); continue

        for model_config in MODELS:
            model_name = model_config["name"]
            print(f"  > Tuning {model_name}...")
            
            best_score = float('inf')
            best_result = None

            for i in range(N_ITER_SEARCH):
                params = sample_parameters(model_config["param_space"])
                
                # Store ALL metrics for every fold
                fold_results = [] 
                
                for train_idx, test_idx in kf.split(X, y):
                    X_tr, X_te = pl.from_pandas(X.iloc[train_idx]), pl.from_pandas(X.iloc[test_idx])
                    y_tr, y_te = y[train_idx], y[test_idx]
                    try:
                        m = run_single_fit(model_config["class"], params, model_config.get("number_axis", 1), X_tr, y_tr, X_te, y_te)
                        fold_results.append(m)
                    except Exception as e:
                        # print(f"Fold failed: {e}") 
                        pass
                
                if fold_results:
                    # Calculate average RMSE to decide if this is the best model
                    avg_rmse = np.mean([res['rmse'] for res in fold_results])
                    
                    if avg_rmse < best_score:
                        best_score = avg_rmse
                        
                        # Calculate averages for ALL metrics (Time, R2, MAE)
                        avg_metrics = {
                            k: float(np.mean([res[k] for res in fold_results])) 
                            for k in fold_results[0].keys()
                        }
                        
                        best_result = {
                            "model": model_name, 
                            "dataset": dataset["name"], 
                            "best_params": params,
                            "noisy_flag": dataset.get("noisy", False),
                            **avg_metrics # Unpacks: rmse, r2_score, mae, training_time
                        }

            if best_result:
                print(f"    -> WINNER: RMSE {best_result['rmse']:.4f} | Time: {best_result['training_time']:.2f}s")
                results.append(best_result)

    output_path = Path("results/results_large_noisy.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_benchmark()