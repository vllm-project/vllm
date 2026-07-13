#!/bin/bash
# Smoke-test + lm_eval harness for the four gfx1250 models, using the exact
# serve/eval recipes from the per-model scripts in this directory:
#
#   gptoss   <- gpt.sh          + eval_gpt.sh       (gpt-oss-120b-w-mxfp4-a-fp8)
#   dsr1     <- dsr1.sh         + eval_dsr1.sh      (DeepSeek-R1-0528-MXFP4)
#   dsv4f    <- dsr4_accurate.sh+ eval_dsr4.sh      (DeepSeek-V4-Flash)
#   minimax  <- minimax.sh      + eval_minimax.sh   (MiniMax-M3-MXFP4)
#
# For each selected model it: starts vllm serve with that model's env+args
# (server + lm_eval output stream to the terminal), waits for /health, runs the
# model's gsm8k lm_eval, then shuts the server down (freeing the GPU) before the
# next one.
#
# Usage:
#   ./vllm_smoketest.sh                          # run all four, in order
#   ./vllm_smoketest.sh --minimax                # run only minimax
#   ./vllm_smoketest.sh --gptoss /some/other/path# run gptoss from a custom path
#   ./vllm_smoketest.sh --dsr1 --minimax         # run a subset
# Each of --gptoss / --dsr1 / --dsv4f / --minimax takes an OPTIONAL model path.
set -uo pipefail

PORT=8000
CANONICAL_ORDER=(gptoss dsr1 dsv4f minimax)

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
  [dsr1]="VLLM_DISABLE_COMPILE_CACHE=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=0 HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_FP8BMM=0"
  [dsv4f]="HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 VLLM_FORCE_TORCH_BLOCK_FP8=1 VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_GEMM=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0"
  [minimax]="VLLM_DISABLE_COMPILE_CACHE=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=0 HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_FP8BMM=0"
)

# --- serve args (everything except --model/--host/--port, which we add).
#     NOTE: JSON values must contain NO spaces (they ride through unquoted $VAR
#     word-splitting); the dsr1 compilation-config below is space-free. ---
declare -A MODEL_SERVE=(
  [gptoss]="--tensor-parallel-size 1 --gpu_memory_utilization 0.7 --attention-backend TRITON_ATTN"
  [dsr1]="--trust-remote-code --no-enable-prefix-caching --no-enable-chunked-prefill --max-model-len 8192 --dtype auto --tensor-parallel-size 1 --distributed-executor-backend mp --max-num-batched-tokens 8192 --max-num-seqs 32 --gpu-memory-utilization 0.90 --compilation-config {\"pass_config\":{\"fuse_attn_quant\":true,\"eliminate_noops\":true,\"fuse_norm_quant\":true,\"fuse_mla_dual_rms_norm\":false,\"enable_qk_norm_rope_fusion\":false},\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\",\"+quant_fp8\"]}"
  [dsv4f]="--tensor-parallel-size 1 --gpu_memory_utilization 0.7 --kv-cache-dtype fp8 --max-model-len 32768 --moe-backend TRITON_UNFUSED"
  [minimax]="--trust-remote-code --language-model-only --skip-mm-profiling --block-size 128 --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill --max-model-len 32768 --dtype auto --tensor-parallel-size 1 --distributed-executor-backend mp --max-num-batched-tokens 32768 --max-num-seqs 32 --gpu-memory-utilization 0.90 --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice"
)

# --- lm_eval: kind (chat=chat-completions+template, raw=completions),
#     extra model_args, and gen_kwargs (verbatim from each eval_*.sh) ---
declare -A EVAL_KIND=( [gptoss]="chat" [dsr1]="chat" [dsv4f]="raw" [minimax]="chat" )
declare -A EVAL_MARGS=(
  [gptoss]="num_concurrent=64,max_retries=3,tokenized_requests=False,max_length=8192"
  [dsr1]="num_concurrent=64,max_retries=3,tokenized_requests=False,max_length=8192,timeout=3600"
  [dsv4f]="num_concurrent=8,max_retries=3,tokenized_requests=False,max_length=32768,timeout=1200"
  [minimax]="num_concurrent=32,max_retries=3,tokenized_requests=False,max_length=32768,timeout=3600"
)
declare -A EVAL_GEN=(
  [gptoss]="max_gen_toks=4096"
  [dsr1]="max_gen_toks=4096,do_sample=True,temperature=0.6,top_p=0.95"
  [dsv4f]="max_gen_toks=2048,do_sample=True,temperature=1.0,top_p=0.95"
  [minimax]="max_gen_toks=2048,do_sample=True,temperature=1.0,top_p=0.95"
)

usage() {
  cat <<EOF
Usage: ./vllm_smoketest.sh [--gptoss [PATH]] [--dsr1 [PATH]] [--dsv4f [PATH]] [--minimax [PATH]] [--port N] [--list]
  --gptoss  [PATH]   run gpt-oss-120b-w-mxfp4-a-fp8   (optional model-path override)
  --dsr1    [PATH]   run DeepSeek-R1-0528-MXFP4        (optional model-path override)
  --dsv4f    [PATH]   run DeepSeek-V4-Flash             (optional model-path override)
  --minimax [PATH]   run MiniMax-M3-MXFP4              (optional model-path override)
  --port N           server port (default: $PORT)
  --list             list models + default paths and exit
With no model flag, all four run in order: ${CANONICAL_ORDER[*]}
Each model: serve -> gsm8k lm_eval (5-shot, limit 100) -> shutdown. Logs stream to this terminal.
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
    --port) PORT="$2"; shift 2 ;;
    --list) for k in "${CANONICAL_ORDER[@]}"; do printf '%-9s -> %s\n' "$k" "${MODEL_PATHS[$k]}"; done; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done
[[ ${#SELECTED[@]} -eq 0 ]] && SELECTED=("${CANONICAL_ORDER[@]}")

echo "Models: ${SELECTED[*]} | Port: $PORT  (all server + lm_eval logs stream to this terminal)"

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
command -v lm_eval >/dev/null 2>&1 || pip install "lm-eval[api]"

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
  env $env vllm serve --model "$model" --host localhost --port "$PORT" $serve &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 240); do          # up to ~20 min for load
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    ps -p "$server_pid" >/dev/null 2>&1 || { echo "[$key] ERROR: server died (see output above)"; RESULT[$key]="FAIL (server died at load)"; stop_server; return; }
    sleep 5
  done
  [[ $ready -eq 1 ]] || { echo "[$key] ERROR: server not ready (timeout)"; RESULT[$key]="FAIL (server timeout)"; stop_server; return; }
  echo "[$key] server ready."

  local kind="${EVAL_KIND[$key]}"

  # --- lm_eval (gsm8k, 5-shot, limit 100) ---
  echo "[$key] === lm_eval (gsm8k) ==="
  local lm_model base_url tmpl=()
  if [[ "$kind" == "chat" ]]; then
    lm_model="local-chat-completions"; base_url="http://localhost:$PORT/v1/chat/completions"; tmpl=(--apply_chat_template)
  else
    lm_model="local-completions"; base_url="http://localhost:$PORT/v1/completions"
  fi
  if lm_eval --model "$lm_model" \
       --model_args "model=$model,base_url=$base_url,${EVAL_MARGS[$key]}" \
       "${tmpl[@]}" \
       --gen_kwargs "${EVAL_GEN[$key]}" \
       --tasks gsm8k --num_fewshot 5 --limit 100
  then RESULT[$key]="PASS (gsm8k table above)"; else RESULT[$key]="FAIL (lm_eval error)"; fi

  stop_server
}

for key in "${SELECTED[@]}"; do run_model "$key"; done

echo; echo "==================== SUMMARY ===================="
for key in "${SELECTED[@]}"; do printf '%-9s : %s\n' "$key" "${RESULT[$key]:-UNKNOWN}"; done
