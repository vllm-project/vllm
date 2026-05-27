#!/usr/bin/env bash
# Triton vs CDNA-INT8 vs CDNA-INT4 — `vllm bench serve` runner
# Each config gets the same workload; only the server-side flags change.

set -uo pipefail

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
LOG_DIR=/logesh/vllm/bench_serve_logs
mkdir -p "$LOG_DIR"

# Common server args
SERVER_COMMON=(
  --port "$PORT"
  --host 0.0.0.0
  --dtype bfloat16
  --max-model-len 4096
  --gpu-memory-utilization 0.6
  --enforce-eager   # disable torch.compile so all 3 starts are fast & equal
)

# Common bench args
BENCH_COMMON=(
  --backend openai
  --base-url "http://localhost:$PORT"
  --model "$MODEL"
  --dataset-name random
  --num-prompts 100
  --random-input-len 1024
  --random-output-len 128
  --ignore-eos
  --request-rate inf
  --metric-percentiles 50,90,99
)

wait_for_server() {
  local tag=$1
  local log=$2
  echo "[$(date +%T)] [$tag] waiting for server ready..."
  for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
      echo "[$(date +%T)] [$tag] server READY (after ${i}s)"
      return 0
    fi
    if ! kill -0 "$3" 2>/dev/null; then
      echo "[$(date +%T)] [$tag] server died early; tail of $log:"
      tail -40 "$log"
      return 1
    fi
    sleep 1
  done
  echo "[$(date +%T)] [$tag] server did NOT come up in 120s; aborting"
  return 1
}

stop_server() {
  local pid=$1
  kill -INT "$pid" 2>/dev/null || true
  for i in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
}

run_one() {
  local tag=$1; shift            # e.g. "TRITON-int8"
  local env_prefix=$1; shift     # e.g. "" or "VLLM_USE_CDNA_PTH_ATTN=1"
  local kv_args=("$@")           # e.g. --kv-cache-dtype int8_per_token_head --calculate-kv-scales

  local server_log="$LOG_DIR/${tag}_server.log"
  local bench_log="$LOG_DIR/${tag}_bench.log"

  echo "==================================================="
  echo "[$(date +%T)] >>> CONFIG: $tag"
  echo "==================================================="
  echo "Server cmd:  ${env_prefix} vllm serve $MODEL ${SERVER_COMMON[*]} ${kv_args[*]}"
  echo "Server log:  $server_log"

  # Start server in background
  if [[ -n "$env_prefix" ]]; then
    env $env_prefix vllm serve "$MODEL" "${SERVER_COMMON[@]}" "${kv_args[@]}" \
        > "$server_log" 2>&1 &
  else
    vllm serve "$MODEL" "${SERVER_COMMON[@]}" "${kv_args[@]}" \
        > "$server_log" 2>&1 &
  fi
  local server_pid=$!

  if ! wait_for_server "$tag" "$server_log" "$server_pid"; then
    stop_server "$server_pid"
    return 1
  fi

  echo "[$(date +%T)] [$tag] running bench..."
  echo "Bench cmd:   vllm bench serve ${BENCH_COMMON[*]}"
  vllm bench serve "${BENCH_COMMON[@]}" 2>&1 | tee "$bench_log"
  local rc=${PIPESTATUS[0]}

  echo "[$(date +%T)] [$tag] stopping server..."
  stop_server "$server_pid"
  echo "[$(date +%T)] [$tag] done (bench rc=$rc)"
  return $rc
}

# ---- 1. Triton baseline (Triton attn + INT8 KV per-token-head) ----
run_one "01-TRITON-int8" "" \
  --kv-cache-dtype int8_per_token_head \
  --calculate-kv-scales

# ---- 2. CDNA INT8 ----
run_one "02-CDNA-int8" "VLLM_USE_CDNA_PTH_ATTN=1" \
  --kv-cache-dtype int8_per_token_head \
  --calculate-kv-scales

# ---- 3. CDNA INT4 ----
run_one "03-CDNA-int4" "VLLM_USE_CDNA_PTH_ATTN=1" \
  --kv-cache-dtype int4_per_token_head \
  --calculate-kv-scales

echo ""
echo "==================================================="
echo "All three runs complete. Logs in $LOG_DIR/"
ls -la "$LOG_DIR"/
