#!/bin/bash
# Run model evaluation using chap-core
# Note: Uses local chap-core due to GAP-004 (API mismatch with published version)

set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHAP_CORE_DIR="${CHAP_CORE_DIR:-/Users/knutdr/Sources/chap-core}"
DATASET="${1:-/Users/knutdr/Sources/chap_benchmarking/csv_datasets/ewars_weekly.csv}"
OUTPUT_NAME="${2:-eval}"

OUTPUT_DIR="$PROJECT_DIR/disease-model/results"
mkdir -p "$OUTPUT_DIR"

echo "Running evaluation..."
echo "  Dataset: $DATASET"
echo "  Using chap-core from: $CHAP_CORE_DIR"

cd "$CHAP_CORE_DIR"
uv run chap evaluate http://localhost:8080 \
    --dataset-csv "$DATASET" \
    --is-chapkit-model

echo ""
echo "Evaluation complete. Results saved to target/report.csv"
cat target/report.csv 2>/dev/null || echo "No report.csv found"
