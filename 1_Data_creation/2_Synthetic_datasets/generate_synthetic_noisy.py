import numpy as np
import polars as pl
import gstools as gs
import s3fs

# --- 1. Connection & Config ---
fs = s3fs.S3FileSystem(anon=False)
s3_path = 's3://projet-benchmark-spatial-interpolation/data/synthetic'

SMALL_DIM = 100   # 10,000 points
LARGE_DIM = 317   # ~100,000 points (sqrt(100,000) approx 316.22)
SEED = 20170519
NOISE_STD = 0.5   # Standard deviation of the added noise

# Setup the spatial model
model = gs.Matern(dim=2, var=1, len_scale=10)
srf = gs.SRF(model, seed=SEED)

def upload_synthetic_to_s3():
    # We define scenarios. Now we create both clean and noisy versions.
    # Dimensions adjusted: 100x100 = 10k (Small), 317x317 ≈ 100k (Large)
    base_scenarios = {
        "S-G-Sm":  {"type": "grid", "dim": SMALL_DIM},
        "S-G-Lg":  {"type": "grid", "dim": LARGE_DIM},
        "S-NG-Sm": {"type": "nogrid", "dim": SMALL_DIM},
        "S-NG-Lg": {"type": "nogrid", "dim": LARGE_DIM},
    }

    for name, config in base_scenarios.items():
        print(f"Generating {name}...")

        # 1. Generate Signal
        if config["type"] == "grid":
            coords = np.arange(config["dim"])
            field = srf.structured([coords, coords])
            xx, yy = np.meshgrid(coords, coords, indexing='ij')
            x_flat, y_flat, val_flat = xx.flatten(), yy.flatten(), field.flatten()
        else:
            rng = np.random.default_rng(SEED)
            num_points = config["dim"]**2
            x_flat = rng.uniform(0, config["dim"], num_points)
            y_flat = rng.uniform(0, config["dim"], num_points)
            val_flat = srf((x_flat, y_flat))

        # 2. Create DataFrame (Clean)
        df_clean = pl.DataFrame({'x': x_flat, 'y': y_flat, 'val': val_flat})
        
        # 3. Create DataFrame (Noisy)
        # Add Gaussian noise
        rng_noise = np.random.default_rng(SEED + 1)
        noise = rng_noise.normal(loc=0.0, scale=NOISE_STD, size=len(val_flat))
        df_noisy = pl.DataFrame({'x': x_flat, 'y': y_flat, 'val': val_flat + noise})

        # 4. Upload Clean
        uri_clean = f"{s3_path}/{name}.parquet"
        print(f"    Writing Clean -> {uri_clean}")
        with fs.open(uri_clean, mode='wb') as f:
            df_clean.write_parquet(f)

        # 5. Upload Noisy (Suffix -N)
        uri_noisy = f"{s3_path}/{name}-N.parquet"
        print(f"    Writing Noisy -> {uri_noisy}")
        with fs.open(uri_noisy, mode='wb') as f:
            df_noisy.write_parquet(f)

    print("All synthetic datasets (clean & noisy) uploaded to S3.")

if __name__ == "__main__":
    upload_synthetic_to_s3()