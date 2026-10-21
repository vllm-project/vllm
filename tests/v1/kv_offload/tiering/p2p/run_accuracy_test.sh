#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Accuracy test driver for the p2p connector
# (OffloadingConnector + TieringOffloadingSpec + p2p tier).
#
# Mirrors tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh:
# brings up N prefillers + M decoders on the local host, fronts them
# with p2p_connector_proxy.py, then runs the connector-agnostic
# test_accuracy.py (gsm8k via lm_eval) against the proxy.
#
# Knobs (env vars unless flagged otherwise):
#   MODEL_NAMES                 space-separated model list (default: Llama-3.2-1B-Instruct)
#   NUM_PREFILL_INSTANCES       default 1
#   NUM_DECODE_INSTANCES        default 1
#   PREFILLER_TP_SIZE           default 1
#   DECODER_TP_SIZE             default 1
#   DP_EP                       when set, deploy the decoder in DP-EP attention
#                               mode (dp=DECODER_TP_SIZE, tp=1,
#                               --enable-expert-parallel) instead of tensor
#                               parallel; the proxy round-robins decode across
#                               the DP replicas. Mirrors
#                               nixl_integration/run_accuracy_test.sh.
#   GPU_MEMORY_UTILIZATION      default 0.45
#   MAX_MODEL_LEN               default 512
#   PREFILL_BLOCK_SIZE          default 128
#   DECODE_BLOCK_SIZE           default 128
#   CPU_BYTES                   default 209715200 (200 MB)
#   VLLM_SERVE_EXTRA_ARGS       comma-separated extra args for vllm serve
#   --decoder-first             toggle decoder-first proxy mode
#
# Examples:
#   bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh
#   NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=2 \
#       bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh
#   bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh --decoder-first
#   DP_EP=1 DECODER_TP_SIZE=2 \
#       bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh

set -xe

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
DECODER_FIRST="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    --decoder-first)
      DECODER_FIRST="true"
      shift 1
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--decoder-first]"
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_NAMES=${MODEL_NAMES:-}
if [[ -n "$MODEL_NAMES" ]]; then
  # shellcheck disable=SC2206
  MODELS=($MODEL_NAMES)
else
  MODELS=(
      "meta-llama/Llama-3.2-1B-Instruct"
  )
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1}
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
# DP-EP attention mode (see header). When DP_EP is set the decoder uses
# data-parallel replicas (dp=DECODER_TP_SIZE) that the proxy round-robins over.
DP_EP=${DP_EP:-}
if [[ -n "$DP_EP" ]]; then
  DECODER_DP_SIZE=${DECODER_TP_SIZE}
else
  DECODER_DP_SIZE=1
fi
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-128}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-128}
CPU_BYTES=${CPU_BYTES:-209715200}
VLLM_SERVE_EXTRA_ARGS=${VLLM_SERVE_EXTRA_ARGS:-}

# Base ports — per-instance offsets layered on top.
PREFILL_HTTP_BASE=8100
DECODE_HTTP_BASE=8200
PREFILL_PD_BASE=7777
DECODE_PD_BASE=$((PREFILL_PD_BASE + NUM_PREFILL_INSTANCES))
PROXY_PORT=8192
P2P_HOST=127.0.0.1

# ---------------------------------------------------------------------------
# Resolve repo root + venv (works in .venv and /workspace/venv pods)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd -P)}"

if [[ -z "${VLLM_BIN:-}" ]]; then
    if [[ -x "${GIT_ROOT}/.venv/bin/vllm" ]]; then
        VLLM_BIN="${GIT_ROOT}/.venv/bin/vllm"
    elif [[ -x "/workspace/venv/bin/vllm" ]]; then
        VLLM_BIN="/workspace/venv/bin/vllm"
    else
        VLLM_BIN="$(command -v vllm)"
    fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${GIT_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${GIT_ROOT}/.venv/bin/python"
    elif [[ -x "/workspace/venv/bin/python" ]]; then
        PYTHON_BIN="/workspace/venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi
echo "Using vllm: ${VLLM_BIN}"
echo "Using python: ${PYTHON_BIN}"

SMI_BIN=$(command -v nvidia-smi || command -v rocm-smi || echo "")

# Trap SIGINT/SIGTERM/EXIT to kill background jobs.
trap 'kill $(jobs -pr) 2>/dev/null || true' SIGINT SIGTERM EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up any running vLLM / proxy instances..."
  pkill -f "vllm serve" || true
  pkill -f "p2p_connector_proxy.py" || true
  sleep 2
}

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    $SMI_BIN --query-gpu=name --format=csv,noheader | wc -l
  elif [[ "$SMI_BIN" == *"rocm"* ]]; then
    $SMI_BIN -l | grep -c GPU
  else
    echo "1"
  fi
}

# Build the OffloadingConnector kv-transfer-config for a given PD port.
# Mirrors deploy_local.sh:131.
build_kv_config() {
  local pd_port=$1
  printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":%s,"secondary_tiers":[{"type":"p2p","host":"%s","port":%s}]}}' \
    "${CPU_BYTES}" "${P2P_HOST}" "${pd_port}"
}

# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "  prefillers=${NUM_PREFILL_INSTANCES} (tp=${PREFILLER_TP_SIZE})"
  if [[ -n "$DP_EP" ]]; then
    echo "  decoders=${NUM_DECODE_INSTANCES}   (dp-ep=${DECODER_TP_SIZE}, tp=1)"
  else
    echo "  decoders=${NUM_DECODE_INSTANCES}   (tp=${DECODER_TP_SIZE})"
  fi
  echo "  decoder_first=${DECODER_FIRST}"
  echo "================================"

  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  PREFILL_PD_PORTS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()
  DECODE_PD_PORTS=()

  local num_gpus
  num_gpus=$(get_num_gpus)
  local next_gpu=0

  # ---- Prefillers ----
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    local gpu_id=$((i * PREFILLER_TP_SIZE % num_gpus))
    local cuda_devs="${gpu_id}"
    for (( j=1; j < PREFILLER_TP_SIZE; j++ )); do
      cuda_devs="${cuda_devs},$(((gpu_id + j) % num_gpus))"
    done
    next_gpu=$(((gpu_id + PREFILLER_TP_SIZE) % num_gpus))

    local http_port=$((PREFILL_HTTP_BASE + i))
    local pd_port=$((PREFILL_PD_BASE + i))
    local kv_cfg
    kv_cfg=$(build_kv_config "${pd_port}")

    echo "Prefiller $i: gpu=[${cuda_devs}] http=${http_port} pd=${pd_port}"

    BASE_CMD="CUDA_VISIBLE_DEVICES=${cuda_devs} \
    PYTHONHASHSEED=42 \
    ${VLLM_BIN} serve ${model_name} \
    --port ${http_port} \
    --enforce-eager \
    --block-size ${PREFILL_BLOCK_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${PREFILLER_TP_SIZE} \
    --kv-transfer-config '${kv_cfg}'"

    if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
      IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
      for arg in "${extra_args[@]}"; do
        BASE_CMD="${BASE_CMD} $arg"
      done
    fi

    eval "${BASE_CMD} &"

    PREFILL_HOSTS+=("${P2P_HOST}")
    PREFILL_PORTS+=("${http_port}")
    PREFILL_PD_PORTS+=("${pd_port}")
  done

  # ---- Decoders ----
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    local gpu_id=$(((next_gpu + i * DECODER_TP_SIZE) % num_gpus))
    local cuda_devs="${gpu_id}"
    for (( j=1; j < DECODER_TP_SIZE; j++ )); do
      cuda_devs="${cuda_devs},$(((gpu_id + j) % num_gpus))"
    done

    local http_port=$((DECODE_HTTP_BASE + i))
    local pd_port=$((DECODE_PD_BASE + i))
    local kv_cfg
    kv_cfg=$(build_kv_config "${pd_port}")

    echo "Decoder   $i: gpu=[${cuda_devs}] http=${http_port} pd=${pd_port}"

    BASE_CMD="CUDA_VISIBLE_DEVICES=${cuda_devs} \
    PYTHONHASHSEED=42 \
    ${VLLM_BIN} serve ${model_name} \
    --port ${http_port} \
    --enforce-eager \
    --block-size ${DECODE_BLOCK_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --kv-transfer-config '${kv_cfg}'"

    # DP-EP attention mode: data-parallel + expert-parallel decode replicas
    # (dp=DECODER_TP_SIZE, tp=1) instead of tensor parallel. Mirrors
    # nixl_integration/run_accuracy_test.sh.
    if [[ -z "$DP_EP" ]]; then
      BASE_CMD="${BASE_CMD} --tensor-parallel-size ${DECODER_TP_SIZE}"
    else
      echo "DP-EP Attention enabled, deploying decoder with dp=${DECODER_TP_SIZE} and tp=1"
      BASE_CMD="${BASE_CMD} --data-parallel-size ${DECODER_TP_SIZE} \
      --tensor-parallel-size 1 --enable-expert-parallel"
    fi

    if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
      IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
      for arg in "${extra_args[@]}"; do
        BASE_CMD="${BASE_CMD} $arg"
      done
    fi

    eval "${BASE_CMD} &"

    DECODE_HOSTS+=("${P2P_HOST}")
    DECODE_PORTS+=("${http_port}")
    DECODE_PD_PORTS+=("${pd_port}")
  done

  # ---- Wait for HTTP readiness ----
  for port in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $port to start..."
    wait_for_server "$port"
  done
  for port in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $port to start..."
    wait_for_server "$port"
  done

  # ---- Proxy ----
  # The proxy currently advertises a single prefiller PD address to decoders.
  # For the 1xM and matched NxM common cases the first prefiller's PD coords
  # are the right pick; multi-prefiller PD round-robin is a follow-up.
  PROXY_CMD="${PYTHON_BIN} ${SCRIPT_DIR}/p2p_connector_proxy.py \
    --port ${PROXY_PORT} \
    --host ${P2P_HOST} \
    --prefiller-hosts ${PREFILL_HOSTS[*]} \
    --prefiller-ports ${PREFILL_PORTS[*]} \
    --decoder-hosts ${DECODE_HOSTS[*]} \
    --decoder-ports ${DECODE_PORTS[*]} \
    --p2p-connector-host ${P2P_HOST} \
    --p2p-connector-port ${PREFILL_PD_PORTS[0]} \
    --decoder-p2p-connector-host ${P2P_HOST} \
    --decoder-p2p-connector-port ${DECODE_PD_PORTS[0]} \
    --decoder-dp-size ${DECODER_DP_SIZE}"

  if [[ "${DECODER_FIRST}" == "true" ]]; then
    PROXY_CMD="${PROXY_CMD} --decoder-first"
  fi

  echo "Starting proxy: ${PROXY_CMD}"
  eval "${PROXY_CMD} &"

  sleep 5

  # ---- Run accuracy test (reused from nixl_integration) ----
  echo "Running tests for $model_name"
  TEST_MODEL=$model_name "${PYTHON_BIN}" -m pytest -s -x \
    "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_accuracy.py"

  cleanup_instances
  sleep 3
}

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All tests completed!"
