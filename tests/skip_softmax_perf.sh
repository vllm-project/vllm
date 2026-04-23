#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
THRESH_PREFILL="$2"
THRESH_DECODE="$3"
KV_DTYPE="$4"
PORT="$5"

ISLS=(10000 100000)
OSL=500
CONCURRENCIES=(16)
NUM_PROMPTS=200           # adjust for longer runs
RESULTS_DIR="results/perf"
mkdir -p "$RESULTS_DIR"

# Sanitise model name for filenames
MODEL_TAG="${MODEL//\//_}"
THRESH_TAG="thresh-pf${THRESH_PREFILL}-dc${THRESH_DECODE}"

for ISL in "${ISLS[@]}"; do
  for CONC in "${CONCURRENCIES[@]}"; do
    TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_isl-${ISL}_osl-${OSL}_conc-${CONC}"
    echo ">>> $TAG"

    vllm bench serve \
      --model "$MODEL" \
      --backend openai \
      --port "$PORT" \
      --endpoint /v1/completions \
      --dataset-name random \
      --input-len "$ISL" \
      --output-len "$OSL" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$CONC" \
      --ignore-eos \
      --save-result \
      --result-dir "$RESULTS_DIR" \
      --result-filename "${TAG}.json" \
      2>&1 | tee "${RESULTS_DIR}/${TAG}.log"

    echo ""
  done
done
