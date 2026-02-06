import numpy as np
import polars as pl
import gstools as gs
import s3fs

# --- 1. Connection & Config ---
fs = s3fs.S3FileSystem(anon=False)
s3_path = 's3://projet-benchmark-spatial-interpolation/data/synthetic'

# Configuration
LARGE_DIM = 317       # ~100,000 points (sqrt(100,000) approx 316.22)
SEED = 20170519

# Define 3 levels of noise (Low, Medium, High)
# 'std': The standard deviation of the Gaussian noise added
NOISE_SCENARIOS = [
    {"suffix": "N1", "std": 0.5},  # Low noise
    {"suffix": "N2", "std": 1.0},  # Medium noise
    {"suffix": "N3", "std": 2.0}   # High noise
]

# Setup the spatial model (Ground Truth Generator)
model = gs.Matern(dim=2, var=1, len_scale=10)
srf = gs.SRF(model, seed=SEED)

def upload_synthetic_to_s3():
    print(f"Generating base signal (Large Non-Grid: {LARGE_DIM**2} points)...")
    
    # 1. Generate Base Signal (Ground Truth)
    # We use a single RNG state for the coordinates to ensure all 3 datasets
    # share the same spatial locations (allows for direct comparison).
    rng = np.random.default_rng(SEED)
    num_points = LARGE_DIM**2
    
    # Random uniform coordinates (No Grid)
    x_flat = rng.uniform(0, LARGE_DIM, num_points)
    y_flat = rng.uniform(0, LARGE_DIM, num_points)
    
    # Generate spatial field values
    val_clean = srf((x_flat, y_flat))
    
    # Optional: Save the pure clean version if needed for reference
    # df_clean = pl.DataFrame({'x': x_flat, 'y': y_flat, 'val': val_clean})
    # with fs.open(f"{s3_path}/S-NG-Lg-Clean.parquet", mode='wb') as f:
    #     df_clean.write_parquet(f)

    # 2. Generate and Upload Noisy Versions
    for scenario in NOISE_SCENARIOS:
        suffix = scenario["suffix"]
        std = scenario["std"]
        
        print(f"  > Processing scenario {suffix} (Noise Std: {std})...")

        # Generate specific noise for this level
        # We seed with SEED + std to ensure different noise patterns per level
        rng_noise = np.random.default_rng(int(SEED + std * 100)) 
        noise = rng_noise.normal(loc=0.0, scale=std, size=len(val_clean))
        
        # Create Noisy DataFrame
        df_noisy = pl.DataFrame({
            'x': x_flat, 
            'y': y_flat, 
            'val': val_clean + noise
        })

        # Define path: S-NG-Lg (Synthetic-NoGrid-Large) + Noise Level
        file_name = f"S-NG-Lg-{suffix}.parquet"
        uri = f"{s3_path}/{file_name}"

        print(f"    Writing -> {uri}")
        with fs.open(uri, mode='wb') as f:
            df_noisy.write_parquet(f)

    print("\nAll noisy large datasets uploaded to S3.")

if __name__ == "__main__":
    upload_synthetic_to_s3()