#!/bin/bash

# Define S3 buckets
S3_REPORTS="s3://projet-benchmark-spatial-interpolation/reports/"
WORK_DIR="/home/onyxia/work/benchmark_spatial_interpolation"

# ---------------------------------------------------------
# 1. Render Quarto Report
# ---------------------------------------------------------
echo "Step 3: Rendering Quarto report to PDF..."
quarto render "$WORK_DIR/final_report.qmd" --to pdf

# ---------------------------------------------------------
# 2. Upload Report to S3
# ---------------------------------------------------------
echo "Step 4: Uploading final report to S3..."
REPORT_PDF="$WORK_DIR/final_report.pdf"

if [ -f "$REPORT_PDF" ]; then
    aws s3 cp "$REPORT_PDF" "$S3_REPORTS"
    echo "  -> Report successfully uploaded to $S3_REPORTS"
else
    echo "  -> ERROR: PDF report generation failed."
    exit 1
fi

echo "Process Complete: Benchmarks run, results saved, and report uploaded."