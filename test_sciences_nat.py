# %%

# Modules for data manipulation
import polars as pl
from polars import col as c
from datetime import datetime
import time
import random

# Modules for machine learning
import lightgbm
from crospint.interpolation import *
from sklearn.model_selection import train_test_split

import polars as pl
import plotly.express as px

# %%
# Extract data covering the northern Altantic ocean
df = (
    pl.scan_parquet(
        "s3://projet-benchmark-spatial-interpolation/data/real/COPERNICUS/COPERNICUS_parquet/"
    )
    .filter(
        c.latitude.is_between(35, 75)
        & ((c.longitude <= 25) | (c.longitude >= 302))
    )
    .collect()
)

# %%
# Prepare the data
noises = [random.random() for i in range(df.shape[0])]

df2 = (
    df
    .with_columns(noise = pl.Series(noises))
    .with_columns(
        date=c.date.cast(pl.Utf8).str.to_date("%Y%m%d"),
        hours_since_ref=(
            (pl.col("valid_time") - pl.lit(datetime(2025, 1, 1, 0)))
            .dt.total_hours()
        ),
        temperature_celsius=c.t2m - 273.15,
        temperature_celsius_noisy=c.t2m - 273.15 + 3 * (c.noise - 0.5)
    )
    .select("latitude", "longitude", "hours_since_ref", "temperature_celsius", "temperature_celsius_noisy")
)

# %%
# Perform the train-test split
train_val, test = train_test_split(
    df2.filter(
        c.hours_since_ref <= 180
    ),
    test_size=0.15,
    random_state=20230516
)

# Perform the train-validation split
train, val = train_test_split(
    train_val,
    train_size=int(round(train_val.height * (0.75/0.85), 0)),
    random_state=20230516
)

# %%

fig = px.scatter_map(
    (
        df2
        .filter(c.date == pl.lit("2025-01-01").str.to_date(), c.hour == 12)
        .to_pandas()
    ),
    lat="latitude",
    lon="longitude",
    color="t2m",
    color_continuous_scale="Viridis",
    size="t2m",              # optional
    zoom=4,
    height=600,
)

fig.update_layout(mapbox_style="carto-positron")
fig.show(renderer="browser")


# %%

baseline_model = create_model_pipeline(
    presence_coordinates=True,
    presence_date=False
)

# %%
parameters_baseline_model = {
    "coord_rotation__coordinates_names": ("latitude", "longitude"),
    "coord_rotation__number_axis": 1,
    "model__seed": 20230516,
    "model__n_estimators": 10000,
    # Very important: 0.12 is optimal, but 0.2 is much faster
    "model__learning_rate": 0.2,
    "model__num_leaves": 1023,
    # Very important: avoid unbalanced trees by capping depth
    "model__max_depth": 15,
    "model__max_bins": 3000,
    "model__bagging_fraction": 1,
    "model__bagging_freq": 0,
    "model__feature_fraction": 1
}
baseline_model.set_params(
    **parameters_baseline_model
)

# %%
baseline_preprocessor = baseline_model[:-1]

baseline_preprocessor.fit(
    train.select("latitude", "longitude", "hours_since_ref"),
    train.select("temperature_celsius_noisy")
)
train_trans = baseline_preprocessor.transform(train)
val_trans = baseline_preprocessor.transform(val)

# %%
start_time = time.monotonic()
eval_set = [
    (train_trans, train["temperature_celsius_noisy"].to_numpy().ravel()),
    (val_trans, val["temperature_celsius_noisy"].to_numpy().ravel())
]
eval_names = ["Train", "Validation"]
baseline_model[-1].fit(
    train_trans,
    train["temperature_celsius_noisy"].to_numpy().ravel(),
    eval_set=eval_set,
    eval_names=eval_names,
    callbacks=[
        lightgbm.log_evaluation(period=50),
        lightgbm.early_stopping(stopping_rounds=10)
    ]
)
end_time = time.monotonic()
print(f"Training time of the baseline model: {end_time - start_time} seconds")





# %%
full_model = create_model_pipeline(
    presence_coordinates=True,
    presence_date=False
)

# %%
parameters_full_model = {
    "coord_rotation__coordinates_names": ("latitude", "longitude"),
    "coord_rotation__number_axis": 11,
    "model__seed": 20230516,
    "model__n_estimators": 10000,
    # Very important: 0.12 is optimal, but 0.2 is much faster
    "model__learning_rate": 0.2,
    "model__num_leaves": 1023,
    # Very important: avoid unbalanced trees by capping depth
    "model__max_depth": 15,
    "model__max_bins": 3000,
    "model__bagging_fraction": 1,
    "model__bagging_freq": 0,
    "model__feature_fraction": 1
}
full_model.set_params(
    **parameters_full_model
)

# %%
full_preprocessor = full_model[:-1]

full_preprocessor.fit(
    train.select("latitude", "longitude", "hours_since_ref"),
    train.select("temperature_celsius_noisy")
)
train_trans = full_preprocessor.transform(train)
val_trans = full_preprocessor.transform(val)

# %%
start_time = time.monotonic()
eval_set = [
    (train_trans, train["temperature_celsius_noisy"].to_numpy().ravel()),
    (val_trans, val["temperature_celsius_noisy"].to_numpy().ravel())
]
eval_names = ["Train", "Validation"]
full_model[-1].fit(
    train_trans,
    train["temperature_celsius_noisy"].to_numpy().ravel(),
    eval_set=eval_set,
    eval_names=eval_names,
    callbacks=[
        lightgbm.log_evaluation(period=50),
        lightgbm.early_stopping(stopping_rounds=20)
    ]
)
end_time = time.monotonic()
print(f"Training time of the full model: {end_time - start_time} seconds")

# %%
