"""
Benchmark script with Hyperparameter Tuning (Randomized Search) + Cross Validation.

Usage:
    uv run python 2_Model_training/benchmark.py
"""
import json
import time
import gc
import random
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
from utils.migbt import SklearnMIXGBooster
from utils.kriging_wrapper import PyKrigeWrapper
from utils.gam_wrapper import PyGAMWrapper
import sys
sys.path.append('geoRF')
from geoRF import GeoRFRegressor
from utils.idw import IDWRegressor
from utils.functions import AddCoordinatesRotation, ConvertToPandas
from utils.s3 import get_df_from_s3
from utils.patch_treeple import apply_treeple_patch
apply_treeple_patch()
from treeple.ensemble import ObliqueRandomForestRegressor

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Search Configuration ---
N_ITER_SEARCH = 10   # How many different parameter combos to try per model
CV_SPLITS = 3        # Reduced to 3 to save time during search (3 folds * 10 iters = 30 runs/model)
RANDOM_STATE = 42

MODELS = [
    # --- Random Forest ---
    {
        "name": "random_forest",
        "class": RandomForestRegressor,
        "number_axis": 1,
        # 'param_space' defines lists of values to sample from
        "param_space": {
            "n_estimators": [100, 200, 300],
            "max_features": ["sqrt", "log2", 1.0],
            "min_samples_leaf": [1, 3, 5, 10],
            "max_depth": [None, 10, 20, 30],
            "n_jobs": [-1],
            "random_state": [42]
        },
    },
    {
        "name": "random_forest_cr",
        "class": RandomForestRegressor,
        "number_axis": 23,
        "param_space": {
            "n_estimators": [100, 200, 300],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 3, 5],
            "max_depth": [None, 15, 25],
            "n_jobs": [-1],
            "random_state": [42]
        },
    },

    # --- XGBoost ---
    {
        "name": "xgboost",
        "class": xgboost.XGBRegressor,
        "number_axis": 1,
        "param_space": {
            "n_estimators": [200, 500, 800],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8, 10],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "n_jobs": [-1],
            "random_state": [42],
            "objective": ["reg:squarederror"]
        },
    },
    {
        "name": "xgboost_cr",
        "class": xgboost.XGBRegressor,
        "number_axis": 23,
        "param_space": {
            "n_estimators": [200, 500, 800],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8, 10],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "n_jobs": [-1],
            "random_state": [42],
            "objective": ["reg:squarederror"]
        },
    },

    # --- MIXGBoost ---
    {
        "name": "mixgboost",
        "class": SklearnMIXGBooster,
        "number_axis": 1,
        "param_space": {
            "k": [10, 20, 30],
            "lamb": [0.01, 0.05, 0.1],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 300],
            "max_depth": [6, 12],
            "n_jobs": [-1]
        }
    },
    
    # --- Baselines (Simple/Fixed) ---
    {
        "name": "knn_3",
        "class": KNeighborsRegressor,
        "number_axis": 1,
        "param_space": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["distance", "uniform"],
            "n_jobs": [-1]
        }
    },
    {
        "name": "idw_p3",
        "class": IDWRegressor,
        "number_axis": 1,
        "param_space": {
            "power": [1, 2, 3, 4],
            "n_neighbors": [10, 15, 20]
        }
    }
]

SIZE_SMALL = 5_000
SIZE_LARGE = 100_000

DATASETS = [
    # --- Real Datasets ---
    {
        "name": "rgealti",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
        "target_n": SIZE_LARGE,
        "transform": "log"
    },
    {
        "name": "bdalti",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "target_n": SIZE_LARGE,
        "transform": "log"
    },
    {
        "name": "bdalti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "filter_col": "departement",
        "filter_val": "48",
        "target_n": SIZE_SMALL,
        "transform": "log"
    },
    {
        "name": "rgealti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/RGEALTI/RGEALTI_parquet/",
        "filter_col": "departement",
        "filter_val": "48",
        "target_n": SIZE_SMALL,
        "transform": "log"
    },
    # --- Synthetic Datasets ---
    {"name": "S-G-Sm", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Sm.parquet", "target_n": SIZE_SMALL},
    {"name": "S-G-Lg", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-G-Lg.parquet", "target_n": SIZE_LARGE},
    {"name": "S-NG-Sm", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Sm.parquet", "target_n": SIZE_SMALL},
    {"name": "S-NG-Lg", "path": "s3://projet-benchmark-spatial-interpolation/data/synthetic/S-NG-Lg.parquet", "target_n": SIZE_LARGE},
]

METRICS = [
    {"name": "r2_score", "func": r2_score},
    {"name": "rmse", "func": root_mean_squared_error},
    {"name": "mae", "func": mean_absolute_error},
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def sample_parameters(param_space: dict) -> dict:
    """Randomly samples one value for each key in the parameter space."""
    return {k: random.choice(v) for k, v in param_space.items()}

def load_dataset(dataset_config: dict) -> tuple:
    """Load, clean, and sample. Returns Pandas X and Numpy y."""
    ldf = get_df_from_s3(dataset_config["path"])
    target_n = dataset_config.get("target_n", 5_000)

    if "val" in ldf.collect_schema().names():
        ldf = ldf.rename({"val": "value"})

    if "rgealti" in dataset_config["name"]:
        # print(f"  Massive dataset: Slicing first 1,000,000 rows lazily...")
        ldf = ldf.slice(0, 1_000_000)

    ldf = ldf.filter(
        (pl.col("value").is_not_null()) &
        (pl.col("value") > 0) &
        (pl.col("x").is_not_null()) &
        (pl.col("y").is_not_null())
    )

    # print(f"  Streaming data collection to RAM...")
    df = ldf.select(["x", "y", "value"]).collect(engine="streaming")

    if dataset_config.get("transform") == "log":
        df = df.with_columns(pl.col("value").log())

    df = df.filter(
        pl.col("value").is_finite() &
        pl.col("x").is_finite() &
        pl.col("y").is_finite()
    )

    if len(df) >= target_n:
        df = df.sample(n=target_n, seed=RANDOM_STATE)
    else:
        print(f"  WARNING: Only {len(df)} rows available after cleaning.")

    X = df.select(["x", "y"]).to_pandas()
    y = df.select("value").to_numpy().ravel().astype(float)

    del df
    gc.collect()
    return X, y


def run_single_fit(model_class, params, number_axis, X_train_pl, y_train, X_test_pl, y_test):
    """Helper to run one fit/predict cycle."""
    pipeline = Pipeline([
        ("coord_rotation", AddCoordinatesRotation(
            coordinates_names=("x", "y"),
            number_axis=number_axis
        )),
        ("pandas_converter", ConvertToPandas()),
        ("ml_model", model_class(**params)),
    ])

    start = time.perf_counter()
    pipeline.fit(X_train_pl, y_train)
    training_time = time.perf_counter() - start
    
    y_pred = pipeline.predict(X_test_pl)
    metrics = {m["name"]: float(m["func"](y_test, y_pred)) for m in METRICS}
    metrics['training_time'] = round(training_time, 2)
    return metrics


def run_benchmark(models: list, datasets: list) -> dict:
    results = []
    
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset['name']} ===")
        try:
            X, y = load_dataset(dataset)
        except Exception as e:
            print(f"  FAILED load {dataset['name']}: {e}")
            continue

        for model_config in models:
            model_name = model_config["name"]
            
            # Skip heavy models on large data if needed (optional logic)
            # if len(X) > 10000 and "gam" in model_name: continue

            print(f"  > Tuning: {model_name} (Search: {N_ITER_SEARCH} iters, CV: {CV_SPLITS} folds)")
            
            best_model_score = float('inf') # Minimizing RMSE
            best_model_result = None
            
            # --- Hyperparameter Search Loop ---
            for i in range(N_ITER_SEARCH):
                # 1. Sample Parameters
                current_params = sample_parameters(model_config["param_space"])
                
                fold_scores = []
                
                # 2. Cross Validation for this parameter set
                for train_idx, test_idx in kf.split(X, y):
                    X_train_pd, X_test_pd = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Polars conversion fix
                    X_train_pl = pl.from_pandas(X_train_pd)
                    X_test_pl = pl.from_pandas(X_test_pd)
                    
                    try:
                        metrics = run_single_fit(
                            model_config["class"], 
                            current_params, 
                            model_config.get("number_axis", 1),
                            X_train_pl, y_train, X_test_pl, y_test
                        )
                        fold_scores.append(metrics['rmse'])
                    except Exception as e:
                        # print(f"    Fit Error: {e}")
                        pass
                
                # 3. Evaluate average performance of this param set
                if fold_scores:
                    avg_rmse = np.mean(fold_scores)
                    # print(f"    Iter {i+1}: RMSE={avg_rmse:.4f} | Params={current_params}")
                    
                    # 4. Update Best
                    if avg_rmse < best_model_score:
                        best_model_score = avg_rmse
                        best_model_result = {
                            "model": model_name,
                            "dataset": dataset["name"],
                            "best_params": current_params,
                            "rmse": avg_rmse, # Average RMSE across folds
                            "cv_folds": CV_SPLITS,
                            # We can also store R2, Time, etc. from the last fold or average them
                            "r2_score": 0.0, # Placeholder, or calculate average
                        }

            if best_model_result:
                print(f"    -> WINNER: RMSE {best_model_result['rmse']:.4f} | {best_model_result['best_params']}")
                results.append(best_model_result)
            else:
                print("    -> No successful runs.")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {"cv_splits": CV_SPLITS, "n_iter_search": N_ITER_SEARCH},
        "results": results,
    }


def save_results(results: dict, output_dir: str = "results") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    filepath = output_path / "benchmark_tuning_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    benchmark_results = run_benchmark(MODELS, DATASETS)
    save_results(benchmark_results)