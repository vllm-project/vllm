#!/bin/bash
set -euo pipefail

# add model args here
declare -A MODEL_PATHS=(
  [gpt-oss-120b-mxfp4]="/data/amd/gpt-oss-120b-w-mxfp4-a-fp8"
  [DeepSeek-R1-0528-MXFP4]="/data/amd/DeepSeek-R1-0528-MXFP4"
)
declare -A MODEL_SERVE_ARGS=(
  [gpt-oss-120b-mxfp4]="--tensor-parallel-size 1 --gpu_memory_utilization 0.7 --attention-backend TRITON_ATTN"
  [DeepSeek-R1-0528-MXFP4]="--tensor-parallel-size 1 --gpu_memory_utilization 0.9 --dtype auto --no-enable-prefix-caching --disable-uvicorn-access-log --trust-remote-code"
)
declare -A MODEL_ENV=(
  [gpt-oss-120b-mxfp4]="HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0"
  [DeepSeek-R1-0528-MXFP4]="HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0"
)

MODEL_NAME="gpt-oss-120b-mxfp4" # default
MODEL_PATH_OVERRIDE=""         # overrided path
MODEL_SET=0                    # whether --model was explicitly passed
RUN_LM_EVAL=0
PORT=8000

usage() {
  cat <<'EOF'
Usage: ./vllm_smoketest.sh [--model NAME] [--model-path PATH] [--lm-eval] [--port PORT] [--list]
  --model NAME       Registry key to run (default: gpt-oss-120b-mxfp4)
  --model-path PATH  Override model path for this run
  --lm-eval          Run lm_eval after curl check
  --port PORT        Server port (default: 8000)
  --list             List available models
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL_NAME="$2"; MODEL_SET=1; shift 2 ;;
    --model-path) MODEL_PATH_OVERRIDE="$2"; shift 2 ;;
    --lm-eval)    RUN_LM_EVAL=1; shift ;;
    --port)       PORT="$2"; shift 2 ;;
    --list)       for n in "${!MODEL_PATHS[@]}"; do echo "$n -> ${MODEL_PATHS[$n]}"; done; exit 0 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "$MODEL_PATH_OVERRIDE" && $MODEL_SET -eq 0 ]]; then
  echo "ERROR: --model-path requires --model (which model's serve args/env to use)." >&2
  usage; exit 1
fi

MODEL="${MODEL_PATHS[$MODEL_NAME]:-$MODEL_NAME}"
[[ -n "$MODEL_PATH_OVERRIDE" ]] && MODEL="$MODEL_PATH_OVERRIDE"
SERVE_ARGS="${MODEL_SERVE_ARGS[$MODEL_NAME]:-}"
MODEL_ENV_ARGS="${MODEL_ENV[$MODEL_NAME]:-}"
echo "Model: $MODEL | Port: $PORT | lm_eval: $RUN_LM_EVAL"

# temp
pip3 install "fastapi<0.137"

LOG_DIR="$(pwd)/vllm_smoke_test_logs"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

server_pid=""
cleanup() { [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null && wait "$server_pid" 2>/dev/null; true; }
trap cleanup EXIT
trap 'exit 130' INT TERM

# --- start server ---
env $MODEL_ENV_ARGS \
vllm serve --model "$MODEL" --host localhost --port "$PORT" $SERVE_ARGS &
server_pid=$!
echo "Waiting for server (pid $server_pid)..."

until curl -s "http://localhost:$PORT/health" &>/dev/null; do
  ps -p "$server_pid" >/dev/null || { echo "ERROR: server died"; tail -n 50 "$LOG_DIR/server.log"; exit 1; }
  sleep 5
done
echo "Server ready."

# --- check if text was generated ---
echo "=== Curl completion check ==="
http_code=$(curl -s -o "$LOG_DIR/curl_completion.log" -w '%{http_code}' \
  "http://localhost:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"San Francisco is a\",\"max_tokens\":32,\"temperature\":0}")
cat "$LOG_DIR/curl_completion.log"; echo
[[ "$http_code" == "200" ]] || { echo "FAIL: HTTP $http_code" >&2; exit 1; }
grep -q '"text"' "$LOG_DIR/curl_completion.log" || { echo "FAIL: no text in response" >&2; exit 1; }
echo "PASS"

# --- lm_eval ---
if [[ $RUN_LM_EVAL -eq 1 ]]; then
  echo "=== lm_eval ==="
  pip install "lm-eval[api]"
  lm_eval --model local-completions \
    --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 3 --limit 100 2>&1 | tee "$LOG_DIR/lm_eval.log"
fi

echo "Done."
