#!/bin/bash

# Define S3 buckets
S3_RESULTS="s3://projet-benchmark-spatial-interpolation/results/"
S3_REPORTS="s3://projet-benchmark-spatial-interpolation/reports/"
WORK_DIR="/home/onyxia/work/benchmark_spatial_interpolation"

# Ensure results directory exists locally
mkdir -p "$WORK_DIR/results"

# ---------------------------------------------------------
# 1. Run Benchmark SMALL
# ---------------------------------------------------------
echo "Step 1: Running Benchmark on SMALL datasets..."
uv run python 2_Model_training/benchmark_small.py

# Upload small results immediately to S3
if [ -f "$WORK_DIR/results/results_small.json" ]; then
    echo "  -> Uploading results_small.json to S3..."
    aws s3 cp "$WORK_DIR/results/results_small.json" "$S3_RESULTS"
else
    echo "  -> WARNING: results_small.json not found!"
fi

# ---------------------------------------------------------
# 2. Run Benchmark LARGE (and Noisy)
# ---------------------------------------------------------
echo "Step 2: Running Benchmark on LARGE/NOISY datasets..."
uv run python 2_Model_training/benchmark_large.py

# Upload large results immediately to S3
if [ -f "$WORK_DIR/results/results_large_noisy.json" ]; then
    echo "  -> Uploading results_large_noisy.json to S3..."
    aws s3 cp "$WORK_DIR/results/results_large_noisy.json" "$S3_RESULTS"
else
    echo "  -> WARNING: results_large_noisy.json not found!"
fi

# ---------------------------------------------------------
# 3. Render Quarto Report
# ---------------------------------------------------------
echo "Step 3: Rendering Quarto report to PDF..."
quarto render "$WORK_DIR/report_rod.qmd" --to pdf

# ---------------------------------------------------------
# 4. Upload Report to S3
# ---------------------------------------------------------
echo "Step 4: Uploading final report to S3..."
REPORT_PDF="$WORK_DIR/report_rod.pdf"

if [ -f "$REPORT_PDF" ]; then
    aws s3 cp "$REPORT_PDF" "$S3_REPORTS"
    echo "  -> Report successfully uploaded to $S3_REPORTS"
else
    echo "  -> ERROR: PDF report generation failed."
    exit 1
fi

echo "Process Complete: Benchmarks run, results saved, and report uploaded."