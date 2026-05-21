#!/bin/bash
# Upload coverage data for the current Buildkite step.
# Called automatically at the end of each step when COLLECT_COVERAGE=1.
#
# Expects:
#   - .coverage.${BUILDKITE_STEP_KEY} data file from coverage run --append
#   - BUILDKITE_STEP_KEY, BUILDKITE_BUILD_NUMBER env vars
#
# Produces:
#   - coverage_${BUILDKITE_STEP_KEY}.json uploaded as a Buildkite artifact

set -euo pipefail

STEP_KEY="${BUILDKITE_STEP_KEY:-unknown}"
DATA_FILE=".coverage.${STEP_KEY}"
OUTPUT_JSON="coverage_${STEP_KEY}.json"

if [ ! -f "$DATA_FILE" ]; then
    echo "~~~ No coverage data file found ($DATA_FILE), skipping upload"
    exit 0
fi

echo "~~~ :bar_chart: Exporting coverage data for step: ${STEP_KEY}"

coverage json \
    --data-file="$DATA_FILE" \
    -o "$OUTPUT_JSON" \
    --omit='*/tests/*,*/test_*,*/__pycache__/*' \
    2>&1 || {
        echo "Warning: coverage json export failed, skipping"
        exit 0
    }

FILE_COUNT=$(python3 -c "import json; d=json.load(open('$OUTPUT_JSON')); print(len(d.get('files', {})))" 2>/dev/null || echo "?")
echo "Coverage captured ${FILE_COUNT} source files for step ${STEP_KEY}"

buildkite-agent artifact upload "$OUTPUT_JSON" 2>&1 || {
    echo "Warning: artifact upload failed"
    exit 0
}

echo "Uploaded $OUTPUT_JSON"
