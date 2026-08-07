#!/bin/bash
# Sweep vllm bench serve over a list of concurrency levels against an
# already-running server/proxy (single instance, round-robin replica
# proxy, or NIXL disagg proxy), and combine the results into a CSV
# matching the column layout used across experiments/*/*.csv.
#
# This is the perf counterpart of test_disagg_accuracy.py: that script
# checks correctness against a running P/D setup, this one measures
# performance against any running endpoint (single GPU, replicas, or
# disagg proxy) with the same benchmark methodology.
#
# Example (single GPU baseline):
#   ./run_perf_sweep.sh --base-url http://localhost:8500 \
#       --model unsloth/gpt-oss-20b --label singleGPU \
#       --out-dir ../../../../experiments/singleGPU/gpt-oss-20b
#
# Example (disagg proxy):
#   ./run_perf_sweep.sh --base-url http://localhost:8192 \
#       --model unsloth/gpt-oss-20b --label disagg \
#       --out-dir ../../../../experiments/disagg/gpt-oss-20b
#
# Example (round-robin replica proxy):
#   ./run_perf_sweep.sh --base-url http://localhost:8300 \
#       --model unsloth/gpt-oss-20b --label replicas \
#       --out-dir ../../../../experiments/replicas/gpt-oss-20b

set -xe

BASE_URL=""
MODEL=""
LABEL=""
OUT_DIR=""
DATASET_PATH="./ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS=1024
CONCURRENCY_LIST=(1 2 4 8 16 32 64 96 128 256)
HEALTH_PATH="/healthcheck"  # /health for a bare (non-proxied) vllm serve

while [[ $# -gt 0 ]]; do
  case $1 in
    --base-url)
      BASE_URL="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --label)
      LABEL="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --num-prompts)
      NUM_PROMPTS="$2"; shift 2 ;;
    --concurrency-list)
      # space-separated string, e.g. --concurrency-list "1 2 4 8"
      read -r -a CONCURRENCY_LIST <<< "$2"
      shift 2 ;;
    --health-path)
      HEALTH_PATH="$2"; shift 2 ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 --base-url <url> --model <name> --label <label> --out-dir <dir> [--dataset-path <path>] [--num-prompts <n>] [--concurrency-list \"1 2 4 ...\"] [--health-path </health|/healthcheck>]"
      exit 1 ;;
  esac
done

if [[ -z "$BASE_URL" || -z "$MODEL" || -z "$LABEL" || -z "$OUT_DIR" ]]; then
  echo "Missing required argument. --base-url, --model, --label, and --out-dir are all required."
  exit 1
fi

mkdir -p "$OUT_DIR"
RES_LOG="${OUT_DIR}/${LABEL}.res"
: > "$RES_LOG"  # truncate/create

# Confirm the endpoint is actually up before burning hours on a sweep.
if ! curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}${HEALTH_PATH}" | grep -q "200"; then
  echo "Endpoint ${BASE_URL}${HEALTH_PATH} is not responding with 200 - aborting."
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

for CONCURRENCY in "${CONCURRENCY_LIST[@]}"; do
  echo "Running sweep point: concurrency=${CONCURRENCY}" | tee -a "$RES_LOG"

  vllm bench serve \
    --backend vllm \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate inf \
    --max-concurrency "$CONCURRENCY" \
    --save-result \
    --result-dir "$OUT_DIR" \
    --result-filename "${LABEL}_c${CONCURRENCY}.json" \
    2>&1 | tee -a "$RES_LOG"
done

python3 "${SCRIPT_DIR}/perf_result_to_csv.py" \
  --results-dir "$OUT_DIR" \
  --glob "${LABEL}_c*.json" \
  --out "${OUT_DIR}/${LABEL}.csv"

echo "Sweep complete. Log: ${RES_LOG}  CSV: ${OUT_DIR}/${LABEL}.csv"
