"""
benchmark_small.py
Target: Small Datasets (~5,000 points)
Models: ALL (including Kriging, GAM, GeoRF)
Tuning: High intensity (20 iterations, 5 folds)
"""
import json
import time
import gc
import random
import sys
from datetime import datetime
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

N_ITER_SEARCH = 20   # High search budget
CV_SPLITS = 5        # Robust validation
RANDOM_STATE = 42
SIZE_SMALL = 5_000

DATASETS = [
    # --- Real Small ---
    {
        "name": "bdalti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "filter_col": "departement", "filter_val": "48",
        "target_n": SIZE_SMALL, "transform": "log"
    },
    {
        "name": "rgealti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
        "filter_col": "departement", "filter_val": "48",
        "target_n": SIZE_SMALL, "transform": "log"
    },
    # --- Synthetic Small ---
    {"name": "S-G-Sm", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Sm.parquet", "target_n": SIZE_SMALL},
    {"name": "S-NG-Sm", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Sm.parquet", "target_n": SIZE_SMALL},
]

MODELS = [
    # --- Ensembles (RF, XGB, MixGB) ---
    {
        "name": "random_forest",
        "class": RandomForestRegressor,
        "number_axis": 1,
        "param_space": {
            "n_estimators": [100, 200, 300, 500],
            "max_features": ["sqrt", "log2", 1.0],
            "min_samples_leaf": [1, 3, 5],
            "n_jobs": [-1], "random_state": [42]
        }
    },
    {
        "name": "random_forest_cr",
        "class": RandomForestRegressor,
        "number_axis": 23,
        "param_space": {
            "n_estimators": [100, 200, 300],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 3, 5],
            "n_jobs": [-1], "random_state": [42]
        }
    },
    {
        "name": "xgboost",
        "class": xgboost.XGBRegressor,
        "number_axis": 1,
        "param_space": {
            "n_estimators": [200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [4, 6, 8, 10],
            "subsample": [0.7, 0.8, 1.0],
            "n_jobs": [-1], "random_state": [42], "objective": ["reg:squarederror"]
        }
    },
    {
        "name": "xgboost_cr",
        "class": xgboost.XGBRegressor,
        "number_axis": 23,
        "param_space": {
            "n_estimators": [200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8, 10],
            "n_jobs": [-1], "random_state": [42], "objective": ["reg:squarederror"]
        }
    },
    {
        "name": "mixgboost_cr",
        "class": SklearnMIXGBooster,
        "number_axis": 23,
        "param_space": {
            "k": [10, 20, 40],
            "lamb": [0.01, 0.1, 0.5],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 500],
            "max_depth": [6, 10, 12],
            "n_jobs": [-1]
        }
    },
    
    # --- Advanced Trees (Oblique, GeoRF) ---
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

    # --- Geostatistical / GAM (Included only for Small) ---
    {
        "name": "kriging",
        "class": PyKrigeWrapper,
        "number_axis": 1,
        "param_space": {"variogram_model": ["linear", "power", "gaussian", "exponential"], "nlags": [6, 10, 20], "weight": [True, False]}
    },
    {
        "name": "gam_cr",
        "class": PyGAMWrapper,
        "number_axis": 23,
        "param_space": {"n_splines": [15, 25, 50], "lam": [0.1, 0.6, 2.0], "spline_order": [3]}
    },

    # --- Baselines ---
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

    # Filter
    ldf = ldf.filter((pl.col("value").is_not_null()) & (pl.col("value") > 0))
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

            for i in range(N_ITER_SEARCH):
                params = sample_parameters(model_config["param_space"])
                fold_scores = []
                
                for train_idx, test_idx in kf.split(X, y):
                    X_tr, X_te = pl.from_pandas(X.iloc[train_idx]), pl.from_pandas(X.iloc[test_idx])
                    y_tr, y_te = y[train_idx], y[test_idx]
                    try:
                        m = run_single_fit(model_config["class"], params, model_config.get("number_axis", 1), X_tr, y_tr, X_te, y_te)
                        fold_scores.append(m['rmse'])
                    except: pass
                
                if fold_scores:
                    avg_rmse = np.mean(fold_scores)
                    if avg_rmse < best_score:
                        best_score = avg_rmse
                        best_result = {"model": model_name, "dataset": dataset["name"], "rmse": avg_rmse, "best_params": params}

            if best_result:
                print(f"    -> WINNER: RMSE {best_result['rmse']:.4f}")
                results.append(best_result)

    with open("results/results_small.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_benchmark()