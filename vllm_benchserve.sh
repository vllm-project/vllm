#!/bin/bash
# Smoke-test + vllm-bench-serve harness for the four gfx1250 models, using the
# exact serve recipes from the per-model scripts in this directory:
#
#   gptoss   <- gpt.sh           (gpt-oss-120b-w-mxfp4-a-fp8)
#   dsr1     <- dsr1.sh          (DeepSeek-R1-0528-MXFP4)
#   dsv4f    <- dsr4_accurate.sh (DeepSeek-V4-Flash)
#   minimax  <- minimax.sh       (MiniMax-M3-MXFP4)
#
# For each selected model it: starts vllm serve with that model's env+args
# (server + bench output stream to the terminal), waits for /health, then runs
# `vllm bench serve` with random 1k/1k ISL/OSL at concurrencies 1,4,8,32,64,
# then shuts the server down (freeing the GPU) before the next one.
#
# Usage:
#   ./vllm_benchserve.sh                          # run all four, in order
#   ./vllm_benchserve.sh --minimax                # run only minimax
#   ./vllm_benchserve.sh --gptoss /some/other/path# run gptoss from a custom path
#   ./vllm_benchserve.sh --dsr1 --minimax         # run a subset
# Each of --gptoss / --dsr1 / --dsv4f / --minimax takes an OPTIONAL model path.
set -uo pipefail

PORT=8000
CANONICAL_ORDER=(gptoss dsr1 dsv4f minimax)

# --- bench-serve parameters ---
INPUT_LEN=1024
OUTPUT_LEN=1024
CONCURRENCIES=(64)                 # default: concurrency 1 only (--long for the full sweep)
LONG_CONCURRENCIES=(1 4 8 32 64)  # used when --long is given
# prompts sent per concurrency = concurrency * PROMPTS_PER_CONC
PROMPTS_PER_CONC=10

# --- default model paths (from the per-model serve scripts) ---
declare -A MODEL_PATHS=(
  [gptoss]="/data/models/gpt-oss-120b-w-mxfp4-a-fp8"
  [dsr1]="/data/models/DeepSeek-R1-0528-MXFP4"
  [dsv4f]="/data/models/DeepSeek-V4-Flash"
  [minimax]="/data/models/MiniMax-M3-MXFP4"
)

# --- serve-time environment (verbatim from each serve script) ---
declare -A MODEL_ENV=(
  [gptoss]="HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0"
  [dsr1]="VLLM_AITER_A4W4_BACKEND=triton VLLM_DISABLE_COMPILE_CACHE=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=0 HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_FP8BMM=0"
  [dsv4f]="VLLM_AITER_A4W4_BACKEND=triton VLLM_FORCE_TORCH_BLOCK_FP8=1 VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_GEMM=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0"
  [minimax]="VLLM_AITER_A4W4_BACKEND=triton VLLM_DISABLE_COMPILE_CACHE=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=0 HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_FP8BMM=0"
)

# --- serve args (everything except --model/--host/--port, which we add).
#     NOTE: JSON values must contain NO spaces (they ride through unquoted $VAR
#     word-splitting); the dsr1 compilation-config below is space-free. ---
declare -A MODEL_SERVE=(
  [gptoss]="--tensor-parallel-size 1 --gpu_memory_utilization 0.7 --attention-backend TRITON_ATTN"
  [dsr1]="--trust-remote-code --no-enable-prefix-caching --no-enable-chunked-prefill --max-model-len 8192 --dtype auto --tensor-parallel-size 1 --distributed-executor-backend mp --max-num-batched-tokens 8192 --max-num-seqs 32 --gpu-memory-utilization 0.90 --compilation-config {\"mode\":0,\"pass_config\":{\"fuse_attn_quant\":true,\"eliminate_noops\":true,\"fuse_norm_quant\":true,\"fuse_mla_dual_rms_norm\":false,\"enable_qk_norm_rope_fusion\":false},\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\",\"+quant_fp8\"]}"
  [dsv4f]="--tensor-parallel-size 1 --gpu_memory_utilization 0.7 --kv-cache-dtype fp8 --max-model-len 32768"
  [minimax]="--trust-remote-code --language-model-only --skip-mm-profiling --block-size 128 --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill --max-model-len 32768 --dtype auto --tensor-parallel-size 1 --distributed-executor-backend mp --max-num-batched-tokens 32768 --max-num-seqs 32 --gpu-memory-utilization 0.90 --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice"
)

usage() {
  cat <<EOF
Usage: ./vllm_benchserve.sh [--gptoss [PATH]] [--dsr1 [PATH]] [--dsv4f [PATH]] [--minimax [PATH]] [--long] [--port N] [--list]
  --gptoss  [PATH]   run gpt-oss-120b-w-mxfp4-a-fp8   (optional model-path override)
  --dsr1    [PATH]   run DeepSeek-R1-0528-MXFP4        (optional model-path override)
  --dsv4f    [PATH]   run DeepSeek-V4-Flash             (optional model-path override)
  --minimax [PATH]   run MiniMax-M3-MXFP4              (optional model-path override)
  --long             sweep concurrencies ${LONG_CONCURRENCIES[*]} (default: ${CONCURRENCIES[*]})
  --port N           server port (default: $PORT)
  --list             list models + default paths and exit
With no model flag, all four run in order: ${CANONICAL_ORDER[*]}
Each model: serve -> vllm bench serve (random ${INPUT_LEN}/${OUTPUT_LEN} isl/osl @ conc ${CONCURRENCIES[*]}) -> shutdown.
Logs stream to this terminal.
EOF
}

# --- parse args (each model flag takes an OPTIONAL path) ---
declare -A PATH_OVERRIDE=()
SELECTED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gptoss|--dsr1|--dsv4f|--minimax)
      key="${1#--}"
      SELECTED+=("$key")
      if [[ $# -ge 2 && "$2" != -* ]]; then PATH_OVERRIDE[$key]="$2"; shift 2; else shift; fi
      ;;
    --long) CONCURRENCIES=("${LONG_CONCURRENCIES[@]}"); shift ;;
    --port) PORT="$2"; shift 2 ;;
    --list) for k in "${CANONICAL_ORDER[@]}"; do printf '%-9s -> %s\n' "$k" "${MODEL_PATHS[$k]}"; done; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done
[[ ${#SELECTED[@]} -eq 0 ]] && SELECTED=("${CANONICAL_ORDER[@]}")

echo "Models: ${SELECTED[*]} | Port: $PORT  (all server + bench logs stream to this terminal)"

server_pid=""
stop_server() {
  [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
  # vLLM renames workers to VLLM::EngineCore / VLLM::Worker; killing the launcher
  # alone leaves them holding GPU memory, so sweep them too.
  pkill -9 -f "vllm serve" 2>/dev/null
  pkill -9 -f "VLLM::"     2>/dev/null
  server_pid=""
  # wait for GPU0 to actually release before the next model loads
  if command -v rocm-smi >/dev/null 2>&1; then
    for _ in $(seq 1 40); do
      used=$(rocm-smi --showmeminfo vram 2>/dev/null | grep 'GPU\[0\]' | grep -i used | grep -oE '[0-9]+' | tail -1)
      [[ -z "$used" ]] && break
      (( used < 5000000000 )) && break   # < ~5 GiB => free
      sleep 3
    done
  else
    sleep 8
  fi
}
trap 'stop_server; exit 130' INT TERM
trap 'stop_server' EXIT

declare -A RESULT

run_model() {
  local key="$1"
  local model="${PATH_OVERRIDE[$key]:-${MODEL_PATHS[$key]}}"
  local env="${MODEL_ENV[$key]}" serve="${MODEL_SERVE[$key]}"

  echo; echo "==================== $key : $model ===================="
  if [[ ! -e "$model" ]]; then
    echo "WARNING: skipping '$key' -- model path does not exist:" >&2
    echo "           $model" >&2
    if [[ -n "${PATH_OVERRIDE[$key]:-}" ]]; then
      echo "         (this path was supplied via '--$key $model'; check the path/mount)" >&2
    else
      echo "         (this is the default path for '$key'; download the model there," >&2
      echo "          or point it elsewhere with:  --$key /path/to/model)" >&2
    fi
    echo "         Continuing with the remaining models." >&2
    RESULT[$key]="SKIP (model path not found: $model)"
    return
  fi

  # --- start server (logs stream to this terminal) ---
  echo "[$key] starting vllm serve ..."
  ( export $env; vllm serve --model "$model" --host localhost --port "$PORT" $serve ) &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 240); do          # up to ~20 min for load
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    ps -p "$server_pid" >/dev/null 2>&1 || { echo "[$key] ERROR: server died (see output above)"; RESULT[$key]="FAIL (server died at load)"; stop_server; return; }
    sleep 5
  done
  [[ $ready -eq 1 ]] || { echo "[$key] ERROR: server not ready (timeout)"; RESULT[$key]="FAIL (server timeout)"; stop_server; return; }
  echo "[$key] server ready."

  # --- vllm bench serve (random ISL/OSL, sweep concurrencies) ---
  local rc=0
  for conc in "${CONCURRENCIES[@]}"; do
    local num_prompts=$(( conc * PROMPTS_PER_CONC ))
    echo "[$key] === vllm bench serve : conc=$conc isl=$INPUT_LEN osl=$OUTPUT_LEN num_prompts=$num_prompts ==="
    if ! vllm bench serve \
         --model "$model" \
         --host localhost --port "$PORT" \
         --dataset-name random \
         --random-input-len "$INPUT_LEN" \
         --random-output-len "$OUTPUT_LEN" \
         --max-concurrency "$conc" \
         --num-prompts "$num_prompts" \
         --ignore-eos
    then
      echo "[$key] ERROR: bench serve failed at conc=$conc" >&2
      rc=1
    fi
  done
  if [[ $rc -eq 0 ]]; then RESULT[$key]="PASS (bench tables above)"; else RESULT[$key]="FAIL (bench serve error)"; fi

  stop_server
}

for key in "${SELECTED[@]}"; do run_model "$key"; done

echo; echo "==================== SUMMARY ===================="
for key in "${SELECTED[@]}"; do printf '%-9s : %s\n' "$key" "${RESULT[$key]:-UNKNOWN}"; done
