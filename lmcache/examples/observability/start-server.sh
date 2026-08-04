#!/bin/bash
# Start LMCache MP server + vLLM with OTLP tracing enabled.
#
# Overridable env vars:
#   MODEL            model path
#   GPU              CUDA device id       (default: 0)
#   VLLM_PORT        vLLM serve port      (default: 8100)
#   LMCACHE_MP_PORT  LMCache server port  (default: 6567)
#   MAX_MODEL_LEN                         (default: 131072)
#   GPU_MEM_UTIL                          (default: 0.5)
#   TENSOR_PARALLEL                       (default: 1)
#   OTLP_ENDPOINT    OTel collector gRPC  (default: http://localhost:4320)

set -euo pipefail

BGPIDS=()
cleanup() {
  echo "Shutting down..."
  for pid in "${BGPIDS[@]}"; do kill "$pid" 2>/dev/null; done
  wait 2>/dev/null
}
trap cleanup INT TERM EXIT

MODEL="${MODEL:-/.cache/huggingface/hub/model}"
GPU="${GPU:-0}"
VLLM_PORT="${VLLM_PORT:-8100}"
LMCACHE_MP_PORT="${LMCACHE_MP_PORT:-6567}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
OTLP_ENDPOINT="${OTLP_ENDPOINT:-http://localhost:4320}"
 

echo "=== LMCache + vLLM with OTLP tracing ==="
echo "  Model:          ${MODEL}"
echo "  GPU:            ${GPU}  vLLM port: ${VLLM_PORT}"
echo "  LMCache port:   ${LMCACHE_MP_PORT}"
echo "  OTLP endpoint:  ${OTLP_ENDPOINT}"

# ---------------------------------------------------------------------------
# 1. LMCache server
# ---------------------------------------------------------------------------
OTEL_SERVICE_NAME=lmcache \
  lmcache server \
  --port "${LMCACHE_MP_PORT}" \
  --l1-size 20 \
  --eviction-policy LRU \
  --chunk-size 256 \
  --otlp-endpoint "${OTLP_ENDPOINT}" \
  --enable-tracing \
  2>&1 | sed 's/^/[lmcache] /' &
BGPIDS+=($!)
sleep 5

# ---------------------------------------------------------------------------
# 2. vLLM
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${GPU}" \
  vllm serve "${MODEL}" \
  --trust-remote-code \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN}" \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-enable-prefix-caching \
  --kv-transfer-config \
    "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":${LMCACHE_MP_PORT}}}" \
  2>&1 | sed 's/^/[vllm] /' &
BGPIDS+=($!)

echo ""
echo "=== Ready ==="
echo "  vLLM:    http://localhost:${VLLM_PORT}"
echo "  Grafana: http://localhost:3000"
echo "  Press Ctrl+C to shut down."

wait
