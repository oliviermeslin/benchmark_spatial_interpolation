import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl
from polars import col as c

try:
    from sklearnex import patch_sklearn  #AOAOAOAO
    patch_sklearn()
except ImportError:
    pass

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils.functions import AddCoordinatesRotation, ConvertToPandas
from utils.s3 import get_df_from_s3

from migbt import SklearnMIXGBooster  #AOAOAOAO


MODELS = [
    # Light 
    {
        "name": "mixgboost_light",
        "class": SklearnMIXGBooster,
        "params": {
            "k": 20,
            "lamb": 0.05,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "max_depth": 12,
            "n_jobs": -1
        },
    },
    # Balanced
    {
        "name": "mixgboost_medium",
        "class": SklearnMIXGBooster,
        "params": {
            "k": 30,
            "lamb": 0.2,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "max_depth": 10,
            "n_jobs": -1
        },
    },
    # Precised
    {
        "name": "mixgboost_precise",
        "class": SklearnMIXGBooster,
        "params": {
            "k": 30,
            "lamb": 0.1,
            "learning_rate": 0.05,
            "n_estimators": 600,
            "max_depth": 10,
            "n_jobs": -1
        },
    }, 
    # Pro
    {
        "name": "mixgboost_ultra",
        "class": SklearnMIXGBooster,
        "params": {
            "k": 20,
            "lamb": 0.01,
            "n_estimators": 200,
            "max_depth": 15,
            "learning_rate": 0.3,
            "n_jobs": -1,
            "min_child_weight": 1,
            "subsample": 1,
            "colsample_bytree": 1,
        },
    },
]

#Dataset

DATASETS = [
    {
        "name": "bdalti",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "sample": 0.005,
        "transform": "log"
    },
    {
        "name": "bdalti_48",
        "path": "s3://projet-benchmark-spatial-interpolation/data/real/BDALTI/BDALTI_parquet/",
        "filter_col": "departement",
        "filter_val": "48",
        "sample": 0.4,
        "transform": "log"
    },
]

METRICS = [
    {"name": "r2_score", "func": r2_score},
    {"name": "rmse", "func": root_mean_squared_error},
    {"name": "mae", "func": mean_absolute_error},
]

# Settings 
TEST_SIZE = 0.5
RANDOM_STATE = 123456



def load_dataset(dataset_config: dict) -> tuple:
    """Load and preprocess a dataset from S3."""
    print(f"  -> Scaricamento dati da: {dataset_config['path']}...")
    ldf = get_df_from_s3(dataset_config["path"])

    if "filter_col" in dataset_config:
        ldf = ldf.filter(pl.col(dataset_config["filter_col"]) == dataset_config["filter_val"])

    df = (
        ldf
        .filter(~c.value.is_nan())
        .filter(c.value > 0)
        .select("x", "y", "value")
        .collect()
    )

    if "sample" in dataset_config:
        if isinstance(dataset_config["sample"], float):
            df = df.sample(fraction=dataset_config["sample"], seed=20230516)
        elif isinstance(dataset_config["sample"], int):
            df = df.sample(n=dataset_config["sample"], seed=20230516)

    X = df.select("x", "y")
    y = df.select("value").to_numpy().ravel()

    if "transform" in dataset_config:
        if dataset_config["transform"] == "log":
            y = np.log(y)

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def run_model(model_config: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate a single model."""
    
    model = model_config["class"](**model_config["params"])

    pipeline = Pipeline(
        [
            ("pandas_converter", ConvertToPandas()),
            ("ml_model", model),
        ]
    )

    print(f"    Inizio training {model_config['name']}...")
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - start
    print(f"    Training completato in {training_time:.2f}s")

    y_pred = pipeline.predict(X_test)
    metrics = {m["name"]: float(m["func"](y_test, y_pred)) for m in METRICS}

    return {
        "model": model_config["name"],
        "params": model_config["params"],
        "training_time": round(training_time, 2),
        **metrics,
    }


def run_benchmark(models: list, datasets: list) -> dict:
    results = []

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset['name']} ===")
        X_train, X_test, y_train, y_test = load_dataset(dataset)
        print(f"  Train size: {len(y_train)}, Test size: {len(y_test)}")

        for model in models:
            try:
                result = run_model(model, X_train, X_test, y_train, y_test)
                result["dataset"] = dataset["name"]
                results.append(result)
                print(f"    [RISULTATO] R2: {result['r2_score']:.4f} | RMSE: {result['rmse']:.4f}")
            except Exception as e:
                print(f"    [ERRORE] Il modello {model['name']} ha fallito: {e}")
                import traceback
                traceback.print_exc()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {"test_size": TEST_SIZE},
        "results": results,
    }


def save_results(results: dict, output_dir: str = "results") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    timestamp = results["timestamp"][:10]
    filepath = output_path / f"benchmark_MATTEO_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSalvataggio completato: {filepath}")
    return filepath



if __name__ == "__main__":
    print("--- AVVIO BENCHMARK PERSONALIZZATO (MI-GBT) ---")
    results = run_benchmark(MODELS, DATASETS)
    save_results(results)