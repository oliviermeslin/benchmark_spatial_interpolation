import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl
from polars import col as c

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Intel Sklearn Extension attiva.")
except ImportError:
    print("Intel Extension non trovata (usando standard sklearn).")

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor 

from utils.s3 import get_df_from_s3

def make_idw_weights(power):
   
    def _weights(distances):
        d = np.maximum(distances, 1e-10)
        return 1.0 / (d ** power)
    return _weights


COMMON_PARAMS = {
    "n_neighbors": 15,      
    "algorithm": "kd_tree", 
    "leaf_size": 40,        
    "n_jobs": -1
}

MODELS = [
    {
        "name": "idw_linear_p1",
        "class": KNeighborsRegressor,
        "params": {
            **COMMON_PARAMS,
            "weights": "distance" 
        },
    },
    
    {
        "name": "idw_gravity_p2",
        "class": KNeighborsRegressor,
        "params": {
            **COMMON_PARAMS,
            "weights": make_idw_weights(power=2)
        },
    },

    {
        "name": "idw_local_p3",
        "class": KNeighborsRegressor,
        "params": {
            **COMMON_PARAMS,
            "weights": make_idw_weights(power=3)
        },
    },
]

# Dataset
DATASETS = [
    {
        "name": "bdalti",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "sample": 0.005, 
        "transform": "log"
    }
]

METRICS = [
    {"name": "r2", "func": r2_score},
    {"name": "rmse", "func": root_mean_squared_error},
    {"name": "mae", "func": mean_absolute_error}, 
]

TEST_SIZE = 0.5
RANDOM_STATE = 123456

def load_dataset(dataset_config):
    print(f"Dataset: {dataset_config['name']}...", flush=True)
    ldf = get_df_from_s3(dataset_config["path"])
    
    if "filter_col" in dataset_config:
        ldf = ldf.filter(pl.col(dataset_config["filter_col"]) == dataset_config["filter_val"])
        
    df = (ldf.filter(~c.value.is_nan()).filter(c.value > 0)
          .select("x", "y", "value").collect())
          
    if "sample" in dataset_config:
        seed = 20230516
        if isinstance(dataset_config["sample"], float):
            df = df.sample(fraction=dataset_config["sample"], seed=seed)
        else:
            df = df.sample(n=dataset_config["sample"], seed=seed)
            
  
    X = df.select("x", "y").to_numpy().astype(np.float32)
    y = df.select("value").to_numpy().ravel().astype(np.float32)
    
    if dataset_config.get("transform") == "log":
        y = np.log(y)
        
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def run_benchmark():
    results = []
    
    for dataset in DATASETS:
        X_train, X_test, y_train, y_test = load_dataset(dataset)
        print(f"  Dati pronti: {len(y_train)} campioni.")
        
        for model_cfg in MODELS:
            print(f"  Running {model_cfg['name']}...", end=" ", flush=True)
            
            # Setup
            model = model_cfg["class"](**model_cfg["params"])
            
            start = time.perf_counter()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            dt = time.perf_counter() - start
            
            print(f"Fatto in {dt:.2f}s")
            
            res = {
                "model": model_cfg["name"],
                "dataset": dataset["name"],
                "time": dt
            }
            # Metrics
            msg = "    -> "
            for m in METRICS:
                val = m["func"](y_test, y_pred)
                res[m["name"]] = val
                msg += f"{m['name']}: {val:.4f} | "
            print(msg)
            
            results.append(res)
            
    # Save
    Path("results").mkdir(exist_ok=True)
    out_file = f"results/benchmark_idw_optimized_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSalvato in: {out_file}")

if __name__ == "__main__":
    run_benchmark()