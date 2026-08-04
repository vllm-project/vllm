#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Explicitly unset PROMETHEUS_MULTIPROC_DIR to let Dynamo manage it internally
unset PROMETHEUS_MULTIPROC_DIR

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# KV cache cap for predictable launches; profiler/test framework overrides via
# env. Applied per worker process (prefill and decode each get the same value,
# on their own GPU).
: "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:=1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"

print_launch_banner "Launching Disaggregated Serving (2 GPUs) + LMCache" "$MODEL" "$HTTP_PORT"

# Start the single LMCache mp server (out-of-process cache engine) shared by the
# prefill and decode workers.
lmcache server \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB:-16}" --eviction-policy LRU \
  --port "$LMCACHE_PORT" --http-port "$LMCACHE_HTTP_PORT" &
SERVER_PID=$!

# Wait until the server's HTTP admin endpoint is healthy before launching workers.
for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "LMCache server process died unexpectedly" >&2
    exit 1
  fi
  curl -sf "http://localhost:$LMCACHE_HTTP_PORT/healthcheck" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://localhost:$LMCACHE_HTTP_PORT/healthcheck" >/dev/null 2>&1 || {
  echo "lmcache server failed to become healthy on :$LMCACHE_HTTP_PORT" >&2
  exit 1
}

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# run decode worker on GPU 0
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
  --model "$MODEL" --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_CONCURRENT_SEQS" \
  $GPU_MEM_ARGS \
  --disaggregation-mode decode \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT}}" &

# run prefill worker on GPU 1
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
  --model "$MODEL" --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_CONCURRENT_SEQS" \
  $GPU_MEM_ARGS \
  --disaggregation-mode prefill \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT}}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
