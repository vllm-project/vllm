#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
THRESH_PREFILL="$2"
THRESH_DECODE="$3"
KV_DTYPE="$4"
PORT="$5"

RESULTS_DIR="results/accuracy"
mkdir -p "$RESULTS_DIR"

MODEL_TAG="${MODEL//\//_}"
THRESH_TAG="thresh-pf${THRESH_PREFILL}-dc${THRESH_DECODE}"

# ── GSM8K (5-shot, 250 samples) ──────────────────────────────────
TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_gsm8k"
echo ">>> Accuracy: $TAG"

lm_eval \
  --model local-completions \
  --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,tokenized_requests=False" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size auto \
  --output_path "${RESULTS_DIR}/${TAG}" \
  2>&1 | tee "${RESULTS_DIR}/${TAG}.log"

echo ""

# ── MMLU Pro (5-shot, 250 samples) ───────────────────────────────
TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_mmlu_pro"
echo ">>> Accuracy: $TAG"

lm_eval \
  --model local-completions \
  --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,tokenized_requests=False" \
  --tasks mmlu_pro \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size auto \
  --output_path "${RESULTS_DIR}/${TAG}" \
  2>&1 | tee "${RESULTS_DIR}/${TAG}.log"

echo ""
