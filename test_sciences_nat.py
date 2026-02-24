# %%
import lightgbm
from crospint.interpolation import *


# Modules for data manipulation
import polars as pl
from polars import col as c

# Modules for machine learning
import lightgbm
from sklearn.model_selection import train_test_split





# %%

df = (
    pl.read_parquet(
        "s3://projet-benchmark-spatial-interpolation/data/real/COPERNICUS/COPERNICUS_parquet/"
    )
)


# %%
