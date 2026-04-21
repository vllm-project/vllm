#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
THRESH_PREFILL="$2"
THRESH_DECODE="$3"
KV_DTYPE="$4"
PORT="$5"

# Optional task selector: space-separated list of accuracy task keys to run.
# Supported keys: gsm8k, mmlu_pro, longbench_e
# Default runs all three. Example:
#   SKIP_SOFTMAX_ACCURACY_TASKS="longbench_e" bash tests/skip_softmax_accuracy.sh ...
TASKS="${SKIP_SOFTMAX_ACCURACY_TASKS:-gsm8k mmlu_pro longbench_e}"

RESULTS_DIR="results/accuracy"
mkdir -p "$RESULTS_DIR"

MODEL_TAG="${MODEL//\//_}"
THRESH_TAG="thresh-pf${THRESH_PREFILL}-dc${THRESH_DECODE}"

run_task() {
  local task_key="$1"

  case "$task_key" in
    gsm8k)
      # ── GSM8K (5-shot, 250 samples) ──────────────────────────────
      local TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_gsm8k"
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
      ;;
    mmlu_pro)
      # ── MMLU Pro (5-shot, 250 samples) ───────────────────────────
      local TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_mmlu_pro"
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
      ;;
    longbench_e)
      # ── LongBench-E (0-shot, 13 length-stratified subtasks) ──────
      # LongBench-E is the right accuracy signal for the skip-softmax
      # work because the savings target long-context attention (the
      # perf sweep uses ISL=10k and 100k). The `_e` variant stratifies
      # docs into 0-4k / 4-8k / 8k+ buckets so we see how accuracy
      # holds up as context grows.
      #
      # Requires optional deps (see tests/skip_softmax_test_plan.md):
      #   jieba fuzzywuzzy rouge python-Levenshtein
      local TAG="${MODEL_TAG}_${THRESH_TAG}_kvdtype-${KV_DTYPE}_longbench_e"
      echo ">>> Accuracy: $TAG"
      lm_eval \
        --model local-completions \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,tokenized_requests=False" \
        --tasks longbench_tasks_e \
        --num_fewshot 0 \
        --limit 100 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${TAG}" \
        2>&1 | tee "${RESULTS_DIR}/${TAG}.log"
      echo ""
      ;;
    *)
      echo ">>> WARNING: unknown accuracy task '$task_key' (skipped)" >&2
      ;;
  esac
}

for task_key in $TASKS; do
  run_task "$task_key"
done
