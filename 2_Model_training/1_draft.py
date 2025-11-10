# %%
import os
import s3fs
import utils.s3

# Modules for data manipulation
import polars as pl
from polars import col as c
import numpy as np

# Modules for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# %%
# Connect to S3
fs = s3fs.S3FileSystem(anon=False)  # set anon=True for public buckets

# Define your S3 path
datapath = 's3://projet-benchmark-spatial-interpolation/data'

# Connection information for Polars
storage_options = {
    "aws_endpoint":  "https://"+os.environ['AWS_S3_ENDPOINT'],
    "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "aws_region": os.environ["AWS_DEFAULT_REGION"],
    "aws_token": os.environ["AWS_SESSION_TOKEN"]
}

# %%
temp = s3.get_df_from_s3(f"{datapath}/real/RGEALTI/RGEALTI_parquet/")

