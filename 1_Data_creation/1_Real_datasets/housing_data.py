import polars as pl
import s3fs
from sklearn.datasets import fetch_california_housing

# --- Config ---
fs = s3fs.S3FileSystem(anon=False)
s3_path = 's3://projet-benchmark-spatial-interpolation/data/real/HOUSING'

def upload_housing_data():
    print("Fetching California Housing dataset...")
    # Fetch data (sklearn uses a cached download)
    housing = fetch_california_housing(as_frame=True)
    df_pandas = housing.frame
    
    # Select and rename columns for consistency
    # Longitude -> x, Latitude -> y, MedHouseVal -> val
    df = pl.from_pandas(df_pandas).select([
        pl.col("Longitude").alias("x"),
        pl.col("Latitude").alias("y"),
        pl.col("MedHouseVal").alias("value") # This is in $100,000s
    ])
    
    print(f"Dataset shape: {df.shape}")
    print("Sample:\n", df.head())

    # Write to S3
    target_uri = f"{s3_path}/california_housing.parquet"
    print(f"Writing to {target_uri}...")
    
    with fs.open(target_uri, mode='wb') as f:
        df.write_parquet(f)
        
    print("Upload complete.")

if __name__ == "__main__":
    upload_housing_data()