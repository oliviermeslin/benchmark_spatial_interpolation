#!/bin/bash

# 1. Run the benchmarking Python script
echo "Step 1: Running Model Training and Benchmarking..."
uv run python 2_Model_training/benchmark.py

# 2. Render the Quarto report to PDF
echo "Step 2: Rendering Quarto report to PDF..."
quarto render /home/onyxia/work/benchmark_spatial_interpolation/report.qmd --to pdf

# 3. Register/Upload the report to S3
echo "Step 3: Registering report on S3..."
REPORT_PDF="/home/onyxia/work/benchmark_spatial_interpolation/report.pdf"
S3_DEST="s3://projet-benchmark-spatial-interpolation/reports/"

aws s3 cp "$REPORT_PDF" "$S3_DEST"

echo "Process Complete: Benchmark run and report uploaded."