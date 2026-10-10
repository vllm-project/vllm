#!/bin/bash
set -xe

# E2E: NixlConnector 1P1D on the layer-pruned GLM-5.3-Flash (hybrid KDA +
# DSA-MLA with kpool indexer/tail). Reuses the Mamba prefix-cache test (the
# ~9000-token prompt spans two 4352-token auto blocks) and adds a P-only vs
# P->D logprob consistency check. Needs 2x SM90/SM100 GPUs.

PREFILL_GPU_ID=${PREFILL_GPU_ID:-0}
DECODE_GPU_ID=${DECODE_GPU_ID:-1}
MODEL=${MODEL:-"JaredforReal/GLM-5.3-Flash-4L"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
# fused_recurrent_kda launches grid.z = num_seqs * local KDA heads; 64 heads at TP1
# with the serve default of 1024 seqs exceeds the CUDA limit (65535).
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
VLLM_SERVE_EXTRA_ARGS=${VLLM_SERVE_EXTRA_ARGS:-}

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

trap 'kill $(jobs -pr) 2>/dev/null' SIGINT SIGTERM EXIT

wait_for_server() {
  local port=$1
  timeout 1500 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  pkill -f "vllm serve" || true
  sleep 2
}

cleanup_instances

EXTRA_ARGS=()
if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
  IFS=',' read -r -a EXTRA_ARGS <<< "$VLLM_SERVE_EXTRA_ARGS"
fi

# The 3-read Mamba conv transfer requires the DS conv-state layout. Text-only
# limits skip the 24-block vision tower profiling (minutes of JIT on cold caches).
launch() {
  local gpu=$1 port=$2 side_channel=$3
  CUDA_VISIBLE_DEVICES=$gpu \
  VLLM_SSM_CONV_STATE_LAYOUT=DS \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$side_channel \
  vllm serve "$MODEL" \
    --port "$port" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 4096 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG" \
    "${EXTRA_ARGS[@]}" &
}

PREFILL_PORT=8001
DECODE_PORT=8002
launch "$PREFILL_GPU_ID" $PREFILL_PORT 5559
launch "$DECODE_GPU_ID" $DECODE_PORT 6000
wait_for_server $PREFILL_PORT
wait_for_server $DECODE_PORT

PROXY_PORT=8192
python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port $PROXY_PORT \
  --prefiller-ports $PREFILL_PORT \
  --decoder-ports $DECODE_PORT &
sleep 5

PREFILL_PORT=$PREFILL_PORT DECODE_PORT=$DECODE_PORT PROXY_PORT=$PROXY_PORT \
python3 -m pytest -s -v \
  "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_mamba_prefix_cache.py" \
  "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_hybrid_pd_consistency.py"

cleanup_instances
