#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Accuracy test driver for the p2p connector
# (OffloadingConnector + TieringOffloadingSpec + p2p tier).
#
# Mirrors tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh:
# brings up N prefillers + M decoders on the local host, fronts them
# with p2p_connector_proxy.py, then runs
#   1. p2p_transfer_check.py - asserts the decoders actually pulled KV over
#      the p2p tier (gsm8k alone passes even if the decoder recomputes), and
#   2. the connector-agnostic test_accuracy.py (gsm8k via lm_eval).
# Cleanup only stops processes and removes /dev/shm files this run created,
# and the run fails if another job holds one of its fixed ports.
#
# Knobs (env vars unless flagged otherwise):
#   MODEL_NAMES                 space-separated model list (default: Llama-3.2-1B-Instruct)
#   NUM_PREFILL_INSTANCES       default 1
#   NUM_DECODE_INSTANCES        default 1
#   PREFILLER_TP_SIZE           default 1
#   PREFILLER_DP_SIZE           default 1; >1 deploys the prefiller with
#                               --data-parallel-size and the proxy pins each
#                               prefill to a replica. Needs NUM_PREFILL_INSTANCES=1.
#   DECODER_TP_SIZE             default 1
#   DP_EP                       when set, deploy the decoder in DP-EP attention
#                               mode (dp=DECODER_TP_SIZE, tp=1,
#                               --enable-expert-parallel) instead of tensor
#                               parallel; the proxy round-robins decode across
#                               the DP replicas. A DP prefiller also gets
#                               --enable-expert-parallel.
#   P2P_MODE                    pd (default): decoder gets a remote_prefiller
#                               block; p2p: proxy --p2p, decoder gets a
#                               remote_kv_source block and probes the source.
#   GPU_MEMORY_UTILIZATION      default 0.45
#   MAX_MODEL_LEN               default 2048 (gsm8k 5-shot prompts exceed 512)
#   PREFILL_BLOCK_SIZE          default 128
#   DECODE_BLOCK_SIZE           default 128
#   CPU_BYTES                   default 2147483648 (2 GiB), sized so the
#                               prefiller does not evict KV before decoders
#                               pull it at NUM_CONCURRENT=100
#   CANONICAL_LAYOUT            1 adds "canonical_layout": true to the
#                               offloading config (topology-free host pages,
#                               the supported route for mixed prefiller/decoder TP)
#   ENFORCE_EAGER               default 1
#   CHECK_TRANSFER              default 1; 0 skips p2p_transfer_check.py
#   EXPECT_TRANSFER             default 1; 0 asserts no KV moved (safe fallback)
#   VLLM_SERVE_EXTRA_ARGS       comma-separated extra args for vllm serve
#   FAULT_CASE                  stall | kill | remove: instead of the proxy and
#                               gsm8k, run p2p_fault_check.py, which pulls from
#                               two prefillers straight into the decoder (p2p
#                               mode) and faults prefiller 0. Needs
#                               NUM_PREFILL_INSTANCES=2, NUM_DECODE_INSTANCES=1.
#   PYTHON_BIN / VLLM_BIN       override interpreter / vllm entrypoint
#   --decoder-first             toggle decoder-first proxy mode
#   --attention-backend <name>  pass --attention-backend to every instance
#
# Examples:
#   bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh
#   NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=2 \
#       bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh
#   P2P_MODE=p2p bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh --decoder-first
#   DP_EP=1 DECODER_TP_SIZE=2 \
#       bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh
#   PREFILLER_DP_SIZE=2 bash tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh

set -xe

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
DECODER_FIRST="false"
ATTENTION_BACKEND=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --decoder-first)
      DECODER_FIRST="true"
      shift 1
      ;;
    --attention-backend)
      ATTENTION_BACKEND="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--decoder-first] [--attention-backend <backend>]"
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
PREFILLER_DP_SIZE=${PREFILLER_DP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
# DP-EP attention mode (see header). When DP_EP is set the decoder uses
# data-parallel replicas (dp=DECODER_TP_SIZE) that the proxy round-robins over.
DP_EP=${DP_EP:-}
if [[ -n "$DP_EP" ]]; then
  DECODER_DP_SIZE=${DECODER_TP_SIZE}
else
  DECODER_DP_SIZE=1
fi
P2P_MODE=${P2P_MODE:-pd}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-128}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-128}
CPU_BYTES=${CPU_BYTES:-2147483648}
ENFORCE_EAGER=${ENFORCE_EAGER:-1}
CANONICAL_LAYOUT=${CANONICAL_LAYOUT:-0}
CHECK_TRANSFER=${CHECK_TRANSFER:-1}
VLLM_SERVE_EXTRA_ARGS=${VLLM_SERVE_EXTRA_ARGS:-}
FAULT_CASE=${FAULT_CASE:-}

if [[ -n "$FAULT_CASE" ]]; then
  if [[ "$FAULT_CASE" != "stall" && "$FAULT_CASE" != "kill" && "$FAULT_CASE" != "remove" ]]; then
    echo "FAULT_CASE must be 'stall', 'kill' or 'remove', got '${FAULT_CASE}'"
    exit 1
  fi
  if (( NUM_PREFILL_INSTANCES != 2 || NUM_DECODE_INSTANCES != 1 || PREFILLER_DP_SIZE != 1 )); then
    echo "FAULT_CASE needs NUM_PREFILL_INSTANCES=2, NUM_DECODE_INSTANCES=1, PREFILLER_DP_SIZE=1"
    exit 1
  fi
  # The fault check pulls with remote_kv_source and reads each pull's
  # cached_tokens.
  P2P_MODE=p2p
  VLLM_SERVE_EXTRA_ARGS="${VLLM_SERVE_EXTRA_ARGS:+${VLLM_SERVE_EXTRA_ARGS},}--enable-prompt-tokens-details"
fi

if [[ "$P2P_MODE" != "pd" && "$P2P_MODE" != "p2p" ]]; then
  echo "P2P_MODE must be 'pd' or 'p2p', got '${P2P_MODE}'"
  exit 1
fi
if (( PREFILLER_DP_SIZE > 1 && NUM_PREFILL_INSTANCES > 1 )); then
  # p2p_connector_proxy.py assumes one prefiller instance fronts the DP ranks.
  echo "PREFILLER_DP_SIZE>1 requires NUM_PREFILL_INSTANCES=1"
  exit 1
fi

# Base ports — per-instance offsets layered on top. Each instance's p2p
# control socket binds base + data_parallel_index, so instances are strided by
# PD_PORT_STRIDE to keep DP ranks of neighbouring instances from colliding.
PREFILL_HTTP_BASE=8100
DECODE_HTTP_BASE=8200
PD_PORT_STRIDE=16
PREFILL_PD_BASE=7700
DECODE_PD_BASE=$((PREFILL_PD_BASE + NUM_PREFILL_INSTANCES * PD_PORT_STRIDE))
PROXY_PORT=8192
P2P_HOST=127.0.0.1
# Pin each engine's internal ports to its own window (see nixl_integration).
PREFILLER_INTERNAL_PORT_BASE=${PREFILLER_INTERNAL_PORT_BASE:-20000}
DECODER_INTERNAL_PORT_BASE=${DECODER_INTERNAL_PORT_BASE:-30000}
INTERNAL_PORT_STRIDE=${INTERNAL_PORT_STRIDE:-100}

# ---------------------------------------------------------------------------
# Resolve repo root + interpreter
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd -P)}"

# Pick the first interpreter that can import vllm, so a stale venv in the
# checkout does not shadow a working one.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for cand in "${GIT_ROOT}/venv/bin/python" "${GIT_ROOT}/.venv/bin/python" \
      "/workspace/venv/bin/python" "$(command -v python3 || true)"; do
    if [[ -x "$cand" ]] && "$cand" -c "import vllm" 2>/dev/null; then
      PYTHON_BIN="$cand"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No python interpreter with vllm installed; set PYTHON_BIN"
  exit 1
fi
if [[ -z "${VLLM_BIN:-}" ]]; then
  VLLM_BIN="$(dirname -- "${PYTHON_BIN}")/vllm"
  [[ -x "$VLLM_BIN" ]] || VLLM_BIN="$(command -v vllm)"
fi
echo "Using vllm: ${VLLM_BIN}"
echo "Using python: ${PYTHON_BIN}"

SMI_BIN=$(command -v nvidia-smi || command -v rocm-smi || echo "")

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

wait_for_proxy() {
  timeout 60 bash -c "
    until curl -sf localhost:${PROXY_PORT}/healthcheck > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Start a command in its own session and record the session id, so cleanup
# only touches processes this run started. A session (unlike a process group)
# also holds the DP API servers, which call setpgrp() and can outlive the
# parent while keeping the HTTP port via SO_REUSEPORT. Background jobs of a
# non-interactive shell are not group leaders, so setsid does not fork and $!
# is the session id.
start_in_session() {
  setsid bash -c "$1" &
  SESSION_IDS+=("$!")
}

cleanup_instances() {
  echo "Cleaning up this run's vLLM / proxy instances..."
  # p2p_fault_check.py records the session of a prefiller it restarted.
  if [[ -s "$FAULT_SID_FILE" ]]; then
    mapfile -t -O "${#SESSION_IDS[@]}" SESSION_IDS < "$FAULT_SID_FILE"
  fi
  rm -f "$FAULT_SID_FILE"
  if (( ${#SESSION_IDS[@]} > 0 )); then
    local sids
    sids=$(IFS=,; echo "${SESSION_IDS[*]}")
    pkill -TERM -s "$sids" || true
    timeout 30 bash -c "while pgrep -s '$sids' > /dev/null; do sleep 1; done" \
      || pkill -KILL -s "$sids" || true
    timeout 30 bash -c "while pgrep -s '$sids' > /dev/null; do sleep 1; done" || true
    SESSION_IDS=()
  fi
  # A crash before the workers' mapping barrier leaves the CPU tier's shm
  # file behind. Every instance's engine_id starts with RUN_ID, so this only
  # matches files from this run, whose owners are gone by now.
  rm -f /dev/shm/vllm_offload_"${RUN_ID}"_*
  local port
  for port in $(seq "$PREFILL_HTTP_BASE" $((PREFILL_HTTP_BASE + NUM_PREFILL_INSTANCES - 1))) \
      $(seq "$DECODE_HTTP_BASE" $((DECODE_HTTP_BASE + NUM_DECODE_INSTANCES - 1))) \
      "$PROXY_PORT"; do
    if ! timeout 60 bash -c "while (echo > /dev/tcp/127.0.0.1/${port}) 2>/dev/null; do sleep 1; done"; then
      # Another job owns it; starting on top of it would mix servers.
      echo "Error: port ${port} is in use by a process this run did not start"
      return 1
    fi
  done
}

# Sessions started by this run (see start_in_session) and the prefix of every
# instance's engine_id.
SESSION_IDS=()
RUN_ID="p2p_e2e_$$"
FAULT_SID_FILE="${TMPDIR:-/tmp}/${RUN_ID}_fault_sids"
trap 'cleanup_instances || true' SIGINT SIGTERM EXIT

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    $SMI_BIN --query-gpu=name --format=csv,noheader | wc -l
  elif [[ "$SMI_BIN" == *"rocm"* ]]; then
    $SMI_BIN -l | grep -c GPU
  else
    echo "1"
  fi
}

# Comma-separated list of `count` GPU ids starting at `start`, wrapping
# around the available devices.
gpu_list() {
  local start=$1 count=$2 num_gpus=$3
  local devs="$((start % num_gpus))"
  for (( j=1; j < count; j++ )); do
    devs="${devs},$(((start + j) % num_gpus))"
  done
  echo "$devs"
}

# Build the OffloadingConnector kv-transfer-config for a given PD port.
build_kv_config() {
  local pd_port=$1 name=$2
  local canonical=""
  if [[ "$CANONICAL_LAYOUT" == "1" ]]; then
    canonical=',"canonical_layout":true'
  fi
  # engine_id names the CPU tier's /dev/shm file; the RUN_ID prefix lets
  # cleanup remove only this run's leftovers.
  printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","engine_id":"%s","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":%s%s,"secondary_tiers":[{"type":"p2p","host":"%s","port":%s}]}}' \
    "${RUN_ID}_${name}" "${CPU_BYTES}" "${canonical}" "${P2P_HOST}" "${pd_port}"
}

# Append the knobs shared by prefillers and decoders to BASE_CMD.
append_common_args() {
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    BASE_CMD="${BASE_CMD} --enforce-eager"
  fi
  if [[ -n "$ATTENTION_BACKEND" ]]; then
    BASE_CMD="${BASE_CMD} --attention-backend=${ATTENTION_BACKEND}"
  fi
  if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
    IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
    for arg in "${extra_args[@]}"; do
      BASE_CMD="${BASE_CMD} $arg"
    done
  fi
}

# Send one short request to every DP rank of a server so first-request JIT
# compilation does not land inside the accuracy run's request timeout.
warmup_ranks() {
  local port=$1 dp_size=$2 model=$3
  for (( r=0; r < dp_size; r++ )); do
    echo "Warming up port ${port} dp rank ${r}"
    curl -sf -m 600 "localhost:${port}/v1/completions" \
      -H "Content-Type: application/json" \
      -H "X-data-parallel-rank: ${r}" \
      -d "{\"model\": \"${model}\", \"prompt\": \"warmup\", \"max_tokens\": 1}" \
      > /dev/null
  done
}

# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------
run_tests_for_model() {
  local model_name=$1
  cleanup_instances
  echo "================================"
  echo "Testing model: $model_name"
  echo "  prefillers=${NUM_PREFILL_INSTANCES} (tp=${PREFILLER_TP_SIZE}, dp=${PREFILLER_DP_SIZE})"
  if [[ -n "$DP_EP" ]]; then
    echo "  decoders=${NUM_DECODE_INSTANCES}   (dp-ep=${DECODER_TP_SIZE}, tp=1)"
  else
    echo "  decoders=${NUM_DECODE_INSTANCES}   (tp=${DECODER_TP_SIZE})"
  fi
  echo "  p2p_mode=${P2P_MODE} decoder_first=${DECODER_FIRST}"
  echo "================================"

  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  PREFILL_PD_PORTS=()
  PREFILL_CMDS=()
  PREFILL_SIDS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()

  local num_gpus
  num_gpus=$(get_num_gpus)
  local next_gpu=0
  local prefill_width=$((PREFILLER_TP_SIZE * PREFILLER_DP_SIZE))

  # ---- Prefillers ----
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    local cuda_devs
    cuda_devs=$(gpu_list "$next_gpu" "$prefill_width" "$num_gpus")
    next_gpu=$(((next_gpu + prefill_width) % num_gpus))

    local http_port=$((PREFILL_HTTP_BASE + i))
    local pd_port=$((PREFILL_PD_BASE + i * PD_PORT_STRIDE))
    local internal_port=$((PREFILLER_INTERNAL_PORT_BASE + i * INTERNAL_PORT_STRIDE))
    local kv_cfg
    kv_cfg=$(build_kv_config "${pd_port}" "prefill${i}")

    echo "Prefiller $i: gpu=[${cuda_devs}] http=${http_port} pd=${pd_port}"

    # For non-DP mode pin VLLM_PORT; for DP mode pin VLLM_DP_MASTER_PORT
    # instead to avoid a race between the DP ranks.
    local internal_port_env
    if (( PREFILLER_DP_SIZE > 1 )); then
      internal_port_env="VLLM_DP_MASTER_PORT=${internal_port}"
    else
      internal_port_env="VLLM_PORT=${internal_port}"
    fi

    BASE_CMD="CUDA_VISIBLE_DEVICES=${cuda_devs} \
    ${internal_port_env} \
    PYTHONHASHSEED=42 \
    ${VLLM_BIN} serve ${model_name} \
    --port ${http_port} \
    --block-size ${PREFILL_BLOCK_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${PREFILLER_TP_SIZE} \
    --kv-transfer-config '${kv_cfg}'"

    if (( PREFILLER_DP_SIZE > 1 )); then
      BASE_CMD="${BASE_CMD} --data-parallel-size ${PREFILLER_DP_SIZE}"
      if [[ -n "$DP_EP" ]]; then
        BASE_CMD="${BASE_CMD} --enable-expert-parallel"
      fi
    fi
    append_common_args

    start_in_session "${BASE_CMD}"

    PREFILL_HOSTS+=("${P2P_HOST}")
    PREFILL_PORTS+=("${http_port}")
    PREFILL_PD_PORTS+=("${pd_port}")
    PREFILL_CMDS+=("${BASE_CMD}")
    PREFILL_SIDS+=("${SESSION_IDS[-1]}")
  done

  # ---- Decoders ----
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    local cuda_devs
    cuda_devs=$(gpu_list "$next_gpu" "$DECODER_TP_SIZE" "$num_gpus")
    next_gpu=$(((next_gpu + DECODER_TP_SIZE) % num_gpus))

    local http_port=$((DECODE_HTTP_BASE + i))
    local pd_port=$((DECODE_PD_BASE + i * PD_PORT_STRIDE))
    local internal_port=$((DECODER_INTERNAL_PORT_BASE + i * INTERNAL_PORT_STRIDE))
    local kv_cfg
    kv_cfg=$(build_kv_config "${pd_port}" "decode${i}")

    echo "Decoder   $i: gpu=[${cuda_devs}] http=${http_port} pd=${pd_port}"

    local internal_port_env
    if [[ -n "$DP_EP" ]]; then
      internal_port_env="VLLM_DP_MASTER_PORT=${internal_port}"
    else
      internal_port_env="VLLM_PORT=${internal_port}"
    fi

    BASE_CMD="CUDA_VISIBLE_DEVICES=${cuda_devs} \
    ${internal_port_env} \
    PYTHONHASHSEED=42 \
    ${VLLM_BIN} serve ${model_name} \
    --port ${http_port} \
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
    append_common_args

    start_in_session "${BASE_CMD}"

    DECODE_HOSTS+=("${P2P_HOST}")
    DECODE_PORTS+=("${http_port}")
  done

  # ---- Wait for HTTP readiness, then warm every DP rank ----
  for port in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $port to start..."
    wait_for_server "$port"
    warmup_ranks "$port" "$PREFILLER_DP_SIZE" "$model_name"
  done
  for port in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $port to start..."
    wait_for_server "$port"
    warmup_ranks "$port" "$DECODER_DP_SIZE" "$model_name"
  done

  # ---- Fault slice: pull straight into the decoder, fault prefiller 0 ----
  if [[ -n "$FAULT_CASE" ]]; then
    # Prefiller i: http port, p2p host/port, session id, and the command
    # that restarts it on the same ports.
    local sources
    sources=$("${PYTHON_BIN}" -c 'import json, sys
a = sys.argv[1:]
print(json.dumps([
    {"http": f"http://localhost:{a[i]}", "host": a[i + 2], "port": int(a[i + 4]),
     "sid": int(a[i + 6]), "cmd": a[i + 8]}
    for i in (0, 1)
]))' "${PREFILL_PORTS[@]}" "${PREFILL_HOSTS[@]}" "${PREFILL_PD_PORTS[@]}" \
      "${PREFILL_SIDS[@]}" "${PREFILL_CMDS[@]}")
    echo "Running fault case ${FAULT_CASE} for $model_name"
    TEST_MODEL=$model_name \
    FAULT_CASE=${FAULT_CASE} \
    DECODE_URL="http://localhost:${DECODE_PORTS[0]}" \
    FAULT_SOURCES="${sources}" \
    FAULT_SID_FILE="${FAULT_SID_FILE}" \
      "${PYTHON_BIN}" -m pytest -s -x "${SCRIPT_DIR}/p2p_fault_check.py"
    cleanup_instances
    sleep 3
    return
  fi

  # ---- Proxy ----
  # Each prefiller instance advertises its own PD base port; the proxy adds
  # the DP rank it pinned the prefill to.
  PROXY_CMD="${PYTHON_BIN} ${SCRIPT_DIR}/p2p_connector_proxy.py \
    --port ${PROXY_PORT} \
    --host ${P2P_HOST} \
    --prefiller-hosts ${PREFILL_HOSTS[*]} \
    --prefiller-ports ${PREFILL_PORTS[*]} \
    --decoder-hosts ${DECODE_HOSTS[*]} \
    --decoder-ports ${DECODE_PORTS[*]} \
    --p2p-connector-host ${PREFILL_HOSTS[*]} \
    --p2p-connector-port ${PREFILL_PD_PORTS[*]} \
    --prefiller-dp-size ${PREFILLER_DP_SIZE} \
    --decoder-dp-size ${DECODER_DP_SIZE}"

  if [[ "${DECODER_FIRST}" == "true" ]]; then
    PROXY_CMD="${PROXY_CMD} --decoder-first"
  fi
  if [[ "${P2P_MODE}" == "p2p" ]]; then
    PROXY_CMD="${PROXY_CMD} --p2p"
  fi

  echo "Starting proxy: ${PROXY_CMD}"
  start_in_session "${PROXY_CMD}"
  wait_for_proxy

  # ---- Assert KV moved over the p2p tier ----
  if [[ "$CHECK_TRANSFER" == "1" ]]; then
    local metrics_urls=()
    for port in "${DECODE_PORTS[@]}"; do
      metrics_urls+=("http://localhost:${port}/metrics")
    done
    echo "Checking p2p KV transfer for $model_name"
    TEST_MODEL=$model_name \
    PROXY_URL="http://localhost:${PROXY_PORT}/v1" \
    DECODE_METRICS_URLS="$(IFS=,; echo "${metrics_urls[*]}")" \
    DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE} \
      "${PYTHON_BIN}" -m pytest -s -x "${SCRIPT_DIR}/p2p_transfer_check.py"
  fi

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
