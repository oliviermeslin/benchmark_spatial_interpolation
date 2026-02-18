# %%
import cdsapi
import cfgrib 
import xarray as xr
import polars as pl
import zipfile

# %%
dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure",
        "total_precipitation"
    ],
    "year": ["2025"],
    "month": ["01"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "grib",
    "download_format": "zip"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
# %%

zip_path = "a66b2c1670e510a6cd5ef35962ef1ccf.zip"
extract_to = "./"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_to)

# %%
import pandas as pd
import pyarrow.parquet as pq
ds = xr.open_dataset("data.grib", engine="cfgrib")
# %%
pq.write_table(ds.to_dataframe().reset_index(), "data_pd.parquet")
# %%

df_polars = pl.from_pandas(ds.to_dataframe().reset_index())
print(df_polars.shape)

# %%
print(df_polars.head())


# %%
ds = xr.open_dataset("data.grib", engine="cfgrib")
# %%
ds.to_zarr(
    "data.zarr",
    mode="w"
)
# %%
df = ds.to_dask_dataframe()

# %%
df.to_parquet("data_parquet/", overwrite=True)


# %%
import pandas as pd
import pyarrow.parquet as pq
from polars import col as c
import polars as pl
import xarray as xr
# Open lazily
ds = xr.open_dataset(
    "data.grib",
    engine="cfgrib",
    chunks="auto"
)

# %%

# Extract timestamps
times = ds.time.values

# %%
# Test with one chunk
# data = (
#     pl.from_pandas(
#         ds.sel(time=times[0]).to_dataframe().reset_index()
#         )
#     .with_columns(
#         hour=c.time.dt.hour(),
#         date=c.time.dt.strftime("%Y%m%d")
#     )
# )

# %%
for date in times:
    print(date)
    pq.write_to_dataset(
        (
            pl.from_pandas(
                ds.sel(time=date).to_dataframe().reset_index()
            )
            .with_columns(
                hour=c.time.dt.hour(),
                date=c.time.dt.strftime("%Y%m%d").cast(pl.Int64)
            )
        ),
        "COPERNICUS_parquet/",
        partitioning=["date", "hour"],
        partitioning_flavor="hive"
    )

# %%

mc cp -r COPERNICUS_parquet s3/projet-benchmark-spatial-interpolation/data/real/COPERNICUS/
# %%

import duckdb

# 1. Create a DuckDB connection (in-memory)
con = duckdb.connect()

# %%
# 3. Query a partitioned Parquet dataset on S3
query = """
CREATE OR REPLACE VIEW temp AS
SELECT *
FROM 's3://projet-benchmark-spatial-interpolation/data/real/COPERNICUS/COPERNICUS_parquet/**/*.parquet';
"""

con.execute(query)
# %%

con.execute("SELECT COUNT(*) FROM temp").df()

# %%
con.execute("SELECT * FROM temp LIMIT 10").df()