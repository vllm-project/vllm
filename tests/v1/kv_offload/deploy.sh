#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Deploy a PD-disaggregated vLLM environment (PDConnector or NixlConnector)
# on OC pod(s). Supports single-pod (multiple GPUs) and multi-pod topologies.
#
# Usage:
#   bash deploy.sh --config configs/cluster_pd.env
#   bash deploy.sh --config configs/cluster_nixl.env
#   LOG_LEVEL=DEBUG bash deploy.sh --config configs/cluster_pd.env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Parse --config
# ---------------------------------------------------------------------------
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: bash deploy.sh --config <config_file>" >&2
    echo "Example configs: configs/cluster_pd.env  configs/cluster_nixl.env" >&2
    exit 1
fi

# Source config file — but env vars already set in the caller's environment
# take precedence (e.g. LOG_LEVEL=DEBUG bash deploy.sh overrides config).
# Strategy: only set a variable from the config if it is not already exported.
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue          # skip comments
    [[ -z "${line//[[:space:]]/}" ]] && continue          # skip blank lines
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
        _key="${BASH_REMATCH[1]}"
        _val="${BASH_REMATCH[2]}"
        # Only apply if the variable is not already set in the environment
        [[ -z "${!_key+x}" ]] && export "${_key}=${_val}"
    fi
done < "$CONFIG_FILE"

# ---------------------------------------------------------------------------
# Defaults (config file already set these; shell env overrides both)
# ---------------------------------------------------------------------------
CONNECTOR="${CONNECTOR:-pd_connector}"
PREFILLER_POD="${PREFILLER_POD:-llmd-transport-decoder}"
DECODER_POD="${DECODER_POD:-llmd-transport-decoder}"
PROXY_POD="${PROXY_POD:-${PREFILLER_POD}}"
# GPU assignment: auto-detect when neither is set (use all pod GPUs, even-split
# in single-pod mode). Explicit values (env or config) take precedence —
# resolved in the auto-detect block below after topology is known.
PREFILLER_GPUS_USER_SET=true; [[ -z "${PREFILLER_GPUS+x}" ]] && PREFILLER_GPUS_USER_SET=false
DECODER_GPUS_USER_SET=true;   [[ -z "${DECODER_GPUS+x}"   ]] && DECODER_GPUS_USER_SET=false

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
CPU_BYTES="${CPU_BYTES:-4294967296}"
DECODER_FIRST="${DECODER_FIRST:-false}"
DISABLE_OFFLOADING="${DISABLE_OFFLOADING:-false}"
# P2P mode: pass --p2p to p2p_connector_proxy.py to exercise the new P2P code
# path (no kv_transfer_params on prefill; decode gets a top-level 'p2p' block).
P2P_MODE="${P2P_MODE:-false}"
# Data-parallel size per role. >1 launches that role's API server with
# --data-parallel-size, fronting that many engine replicas; the proxy then
# round-robins requests across the replicas via the X-data-parallel-rank
# header. Default 1 (single replica, unchanged behavior).
PREFILLER_DP="${PREFILLER_DP:-1}"
DECODER_DP="${DECODER_DP:-1}"

VLLM_BIN="${VLLM_BIN:-/workspace/venv/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

PREFILLER_HTTP_PORT="${PREFILLER_HTTP_PORT:-8100}"
DECODER_HTTP_PORT="${DECODER_HTTP_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8192}"
# P2P control-socket port. Left to the vLLM default (VLLM_P2P_SIDE_CHANNEL_PORT);
# the decoder is bumped +1 only when it shares a host with the prefiller (below).
P2P_PORT="${VLLM_P2P_SIDE_CHANNEL_PORT:-5710}"
NIXL_SIDE_CHANNEL_PORT_PREFILLER="${NIXL_SIDE_CHANNEL_PORT_PREFILLER:-5559}"
NIXL_SIDE_CHANNEL_PORT_DECODER="${NIXL_SIDE_CHANNEL_PORT_DECODER:-5659}"

PREFILLER_LOG="${PREFILLER_LOG:-/tmp/prefiller.log}"
DECODER_LOG="${DECODER_LOG:-/tmp/decoder.log}"
PROXY_LOG="${PROXY_LOG:-/tmp/proxy.log}"
STATE_FILE="${STATE_FILE:-/tmp/deploy_state.env}"

# ---------------------------------------------------------------------------
# Topology detection
# ---------------------------------------------------------------------------
if [[ "$PREFILLER_POD" == "$DECODER_POD" ]]; then
    SINGLE_POD=true
else
    SINGLE_POD=false
fi

# ---------------------------------------------------------------------------
# Auto-detect GPUs (if not user-set) and derive tensor-parallel size
# ---------------------------------------------------------------------------
get_pod_gpu_count() {
    oc exec "$1" -- nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
        | grep -c '^[0-9]'
}

if ! $PREFILLER_GPUS_USER_SET && ! $DECODER_GPUS_USER_SET; then
    if $SINGLE_POD; then
        n=$(get_pod_gpu_count "$PREFILLER_POD")
        if (( n < 1 )); then
            echo "ERROR: no GPUs found on pod ${PREFILLER_POD}" >&2
            exit 1
        elif (( n == 1 )); then
            echo "ERROR: single-pod auto-GPU needs >=2 GPUs (found 1 on ${PREFILLER_POD})." >&2
            echo "       Set PREFILLER_GPUS=0 DECODER_GPUS=0 explicitly to share one GPU." >&2
            exit 1
        elif (( n % 2 != 0 )); then
            echo "ERROR: single-pod auto-split needs even GPU count (found ${n} on ${PREFILLER_POD})." >&2
            echo "       Set PREFILLER_GPUS / DECODER_GPUS explicitly." >&2
            exit 1
        else
            half=$(( n / 2 ))
            PREFILLER_GPUS=$(seq -s, 0 $((half - 1)))
            DECODER_GPUS=$(seq -s, "$half" $((n - 1)))
        fi
    else
        np=$(get_pod_gpu_count "$PREFILLER_POD")
        nd=$(get_pod_gpu_count "$DECODER_POD")
        if (( np < 1 || nd < 1 )); then
            echo "ERROR: GPU detect failed (prefiller=${np} decoder=${nd})" >&2
            exit 1
        fi
        PREFILLER_GPUS=$(seq -s, 0 $((np - 1)))
        DECODER_GPUS=$(seq -s, 0 $((nd - 1)))
    fi
elif [[ "$PREFILLER_GPUS_USER_SET" != "$DECODER_GPUS_USER_SET" ]]; then
    echo "ERROR: must set both PREFILLER_GPUS and DECODER_GPUS, or neither." >&2
    exit 1
fi

# Tensor-parallel size = (#GPUs in the assignment) / DP, overridable via env.
# With DP>1 the pod's GPUs are split into DP replicas of TP each, so TP*DP must
# equal the GPU count. Fail loudly if DP does not divide the GPU count.
gpu_count_in() { awk -F, '{print NF}' <<< "$1"; }
check_dp_divides() {  # $1=role $2=gpu_count $3=dp
    (( $3 <= 1 )) && return 0
    (( $2 % $3 == 0 )) || {
        echo "ERROR: ${1} DP=$3 does not evenly divide its $2 GPU(s)." >&2
        echo "       Set ${1^^}_TP explicitly or pick a DP that divides the GPUs." >&2
        exit 1
    }
}
check_dp_divides prefiller "$(gpu_count_in "$PREFILLER_GPUS")" "$PREFILLER_DP"
check_dp_divides decoder   "$(gpu_count_in "$DECODER_GPUS")"   "$DECODER_DP"
PREFILLER_TP="${PREFILLER_TP:-$(( $(gpu_count_in "$PREFILLER_GPUS") / PREFILLER_DP ))}"
DECODER_TP="${DECODER_TP:-$(( $(gpu_count_in "$DECODER_GPUS") / DECODER_DP ))}"

# Data-parallel flags (only emitted for DP>1).
PREFILLER_DP_FLAG=""
(( PREFILLER_DP > 1 )) && PREFILLER_DP_FLAG="--data-parallel-size ${PREFILLER_DP}"
DECODER_DP_FLAG=""
(( DECODER_DP > 1 )) && DECODER_DP_FLAG="--data-parallel-size ${DECODER_DP}"

echo "=== Deploy: ${CONNECTOR} ==="
if [[ "$CONNECTOR" == "nixl" && "$DISABLE_OFFLOADING" != "true" ]]; then
    echo "  Mode:      MultiConnector(NixlConnector + OffloadingConnector)"
fi
echo "  Prefiller: pod=${PREFILLER_POD} gpus=[${PREFILLER_GPUS}] tp=${PREFILLER_TP} dp=${PREFILLER_DP} http=:${PREFILLER_HTTP_PORT}"
echo "  Decoder:   pod=${DECODER_POD} gpus=[${DECODER_GPUS}] tp=${DECODER_TP} dp=${DECODER_DP} http=:${DECODER_HTTP_PORT}"
{ (( PREFILLER_DP > 1 )) || (( DECODER_DP > 1 )); } && echo "  DP round-robin: prefiller x${PREFILLER_DP}, decoder x${DECODER_DP} (proxy round-robins both roles)"
echo "  Proxy:     pod=${PROXY_POD} http=:${PROXY_PORT}"
[[ "$CONNECTOR" == "pd_connector" && "$P2P_MODE" == "true" ]] && echo "  P2P mode:  on (--p2p)"
echo "  Model:     ${MODEL}  gpu_mem=${GPU_MEM_UTIL}  max_len=${MAX_MODEL_LEN}"
echo "  Single-pod: ${SINGLE_POD}"
echo ""

# ---------------------------------------------------------------------------
# Resolve addresses
# In single-pod mode services talk over localhost; multi-pod uses pod IPs.
# ---------------------------------------------------------------------------
get_pod_ip() {
    oc get pod "$1" -o jsonpath='{.status.podIP}'
}

if $SINGLE_POD; then
    PREFILLER_ADDR="127.0.0.1"
    DECODER_ADDR="127.0.0.1"
else
    PREFILLER_ADDR="$(get_pod_ip "${PREFILLER_POD}")"
    DECODER_ADDR="$(get_pod_ip "${DECODER_POD}")"
    echo "  Prefiller IP: ${PREFILLER_ADDR}"
    echo "  Decoder IP:   ${DECODER_ADDR}"
    echo ""
fi

# Same host → move the decoder's P2P base above the prefiller's replica range
# [base .. base+PREFILLER_DP-1] to avoid a bind collision; different hosts →
# both sides use the default base and only the bind host is exported.
if [[ "${PREFILLER_ADDR}" == "${DECODER_ADDR}" ]]; then
    DECODER_P2P_PORT=$((P2P_PORT + PREFILLER_DP))
    DECODER_P2P_PORT_EXPORT="export VLLM_P2P_SIDE_CHANNEL_PORT=${DECODER_P2P_PORT}"
else
    DECODER_P2P_PORT=${P2P_PORT}
    DECODER_P2P_PORT_EXPORT=""
fi

# ---------------------------------------------------------------------------
# Stop existing instances
# ---------------------------------------------------------------------------
# Kill stale processes, free /dev/shm, wait for GPU memory release on a pod.
# In multi-pod mode this runs on each pod; in single-pod mode just once.
cleanup_pod() {
    local pod="$1" http_port="$2" role="$3"
    echo "--- cleanup ${role} pod=${pod} ---"
    # Use separate oc exec per kill so OOM-killing one shell doesn't abort the rest.
    oc exec "${pod}" -- pkill -9 -f "vllm serve.*${http_port}" 2>/dev/null || true
    oc exec "${pod}" -- pkill -9 -f "VLLM::EngineCore"         2>/dev/null || true

    oc exec "${pod}" -- bash -c "
        shm_before=\$(du -s /dev/shm 2>/dev/null | awk '{print \$1}')
        rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-* 2>/dev/null
        shm_after=\$(du -s /dev/shm 2>/dev/null | awk '{print \$1}')
        freed=\$(( (shm_before - shm_after) / 1048576 ))
        [[ \$freed -gt 0 ]] && echo \"Freed \${freed}GB from /dev/shm\" || echo '/dev/shm clean'
    " 2>/dev/null || true

    oc exec "${pod}" -- bash -c "
        echo -n 'Waiting for GPU memory to be released ...'
        deadline=\$(( \$(date +%s) + 60 ))
        while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
            if [[ \$(date +%s) -ge \$deadline ]]; then
                echo ' timeout (proceeding anyway)'
                break
            fi
            sleep 2
            echo -n '.'
        done
        echo ' done'
    "
}

echo "=== Stopping existing instances ==="

# Proxy lives on PROXY_POD — kill it and wait for its port to free.
oc exec "${PROXY_POD}" -- pkill -9 -f "p2p_connector_proxy.py" 2>/dev/null || true
oc exec "${PROXY_POD}" -- pkill -9 -f "toy_proxy_server.py"   2>/dev/null || true
oc exec "${PROXY_POD}" -- bash -c "
    echo -n 'Waiting for proxy port ${PROXY_PORT} to be free ...'
    deadline=\$(( \$(date +%s) + 30 ))
    while ss -tlnp 2>/dev/null | grep -q ':${PROXY_PORT}[[:space:]]'; do
        if [[ \$(date +%s) -ge \$deadline ]]; then
            echo ' timeout (port still in use)' >&2
            exit 1
        fi
        sleep 1
        echo -n '.'
    done
    echo ' free'
"

cleanup_pod "${PREFILLER_POD}" "${PREFILLER_HTTP_PORT}" "prefiller"
if ! $SINGLE_POD; then
    cleanup_pod "${DECODER_POD}" "${DECODER_HTTP_PORT}" "decoder"
fi

# ---------------------------------------------------------------------------
# Copy scripts to pod(s)
# ---------------------------------------------------------------------------
echo "=== Copying scripts ==="
oc cp "${SCRIPT_DIR}/tiering/p2p/p2p_connector_proxy.py" \
    "${PROXY_POD}:/tmp/p2p_connector_proxy.py"
oc cp "${REPO_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
    "${PROXY_POD}:/tmp/toy_proxy_server.py"

# ---------------------------------------------------------------------------
# Build KV transfer configs
# ---------------------------------------------------------------------------
if [[ "${CONNECTOR}" == "pd_connector" ]]; then
    # host/port omitted: the P2P manager resolves them from
    # VLLM_P2P_SIDE_CHANNEL_HOST / VLLM_P2P_SIDE_CHANNEL_PORT exported below.
    PREFILLER_KV_CONFIG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${CPU_BYTES},\"secondary_tiers\":[{\"type\":\"p2p\"}]}}"
    DECODER_KV_CONFIG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${CPU_BYTES},\"secondary_tiers\":[{\"type\":\"p2p\"}]}}"
elif [[ "${CONNECTOR}" == "nixl" ]]; then
    # Bypass NIXL prefill/decode compatibility hash check — needed when the
    # two pods may differ in vLLM version, dtype, or attention backend.
    _NIXL_EXTRA="\"kv_connector_extra_config\":{\"enforce_handshake_compat\":false}"
    if [[ "${DISABLE_OFFLOADING}" == "true" ]]; then
        PREFILLER_KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",${_NIXL_EXTRA}}"
        DECODER_KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",${_NIXL_EXTRA}}"
    else
        _MC="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",${_NIXL_EXTRA}},{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${CPU_BYTES}}}]}}"
        PREFILLER_KV_CONFIG="${_MC}"
        DECODER_KV_CONFIG="${_MC}"
    fi
else
    echo "ERROR: unknown CONNECTOR='${CONNECTOR}' (expected pd_connector|nixl)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Start Prefiller
# ---------------------------------------------------------------------------
# Write launcher scripts to pod — env vars baked in at write time so nohup
# inherits them correctly regardless of the container's default env.
echo "=== Writing launcher scripts ==="
oc exec "${PREFILLER_POD}" -- bash -c "cat > /tmp/start_prefiller.sh << 'LAUNCHER'
#!/usr/bin/env bash
export VLLM_LOGGING_LEVEL=${LOG_LEVEL}
export CUDA_VISIBLE_DEVICES=${PREFILLER_GPUS}
export PYTHONHASHSEED=42
export VLLM_NIXL_SIDE_CHANNEL_HOST=${PREFILLER_ADDR}
export VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_SIDE_CHANNEL_PORT_PREFILLER}
export VLLM_P2P_SIDE_CHANNEL_HOST=${PREFILLER_ADDR}
export UCX_NET_DEVICES=all
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}
cd /tmp
exec ${VLLM_BIN} serve '${MODEL}' \
    --port ${PREFILLER_HTTP_PORT} \
    --tensor-parallel-size ${PREFILLER_TP} \
    ${PREFILLER_DP_FLAG} \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --kv-transfer-config '${PREFILLER_KV_CONFIG}' \
    ${EXTRA_VLLM_ARGS:-}
LAUNCHER
chmod +x /tmp/start_prefiller.sh"

oc exec "${DECODER_POD}" -- bash -c "cat > /tmp/start_decoder.sh << 'LAUNCHER'
#!/usr/bin/env bash
export VLLM_LOGGING_LEVEL=${LOG_LEVEL}
export CUDA_VISIBLE_DEVICES=${DECODER_GPUS}
export PYTHONHASHSEED=42
export VLLM_NIXL_SIDE_CHANNEL_HOST=${DECODER_ADDR}
export VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_SIDE_CHANNEL_PORT_DECODER}
export VLLM_P2P_SIDE_CHANNEL_HOST=${DECODER_ADDR}
${DECODER_P2P_PORT_EXPORT}
export UCX_NET_DEVICES=all
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}
cd /tmp
exec ${VLLM_BIN} serve '${MODEL}' \
    --port ${DECODER_HTTP_PORT} \
    --tensor-parallel-size ${DECODER_TP} \
    ${DECODER_DP_FLAG} \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --kv-transfer-config '${DECODER_KV_CONFIG}' \
    ${EXTRA_VLLM_ARGS:-}
LAUNCHER
chmod +x /tmp/start_decoder.sh"

echo "=== Starting Prefiller ==="
PREFILLER_PID=$(oc exec "${PREFILLER_POD}" -- bash -c "
    nohup /tmp/start_prefiller.sh > ${PREFILLER_LOG} 2>&1 &
    echo \$!
" 2>/dev/null | tail -1)
echo "  PID ${PREFILLER_PID} → ${PREFILLER_LOG}"

# ---------------------------------------------------------------------------
# Start Decoder
# ---------------------------------------------------------------------------
echo "=== Starting Decoder ==="
DECODER_PID=$(oc exec "${DECODER_POD}" -- bash -c "
    nohup /tmp/start_decoder.sh > ${DECODER_LOG} 2>&1 &
    echo \$!
" 2>/dev/null | tail -1)
echo "  PID ${DECODER_PID} → ${DECODER_LOG}"

# ---------------------------------------------------------------------------
# Wait for health
# ---------------------------------------------------------------------------
wait_for_health() {
    local pod="$1" name="$2" addr="$3" port="$4" log="$5" path="${6:-/health}"
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    echo -n "Waiting for ${name} (${addr}:${port}) ..."
    while true; do
        if oc exec "${pod}" -- curl -sf "http://${addr}:${port}${path}" > /dev/null 2>&1; then
            echo " ready"
            return 0
        fi
        if [[ "$(date +%s)" -ge "${deadline}" ]]; then
            echo ""
            echo "ERROR: ${name} did not become healthy within ${HEALTH_TIMEOUT}s" >&2
            echo "--- last 20 lines of ${log} ---" >&2
            oc exec "${pod}" -- tail -20 "${log}" 2>/dev/null >&2 || true
            return 1
        fi
        sleep 5
        echo -n "."
    done
}

wait_for_health "${PREFILLER_POD}" "Prefiller" "${PREFILLER_ADDR}" "${PREFILLER_HTTP_PORT}" "${PREFILLER_LOG}" || exit 1

# Multi-pod connectivity preflight: now that the prefiller HTTP port is
# listening, confirm the decoder pod can actually reach it. If this fails
# the KV transport will fail too — surface the routing/NetworkPolicy issue
# now rather than after a long decoder timeout.
if ! $SINGLE_POD; then
    echo -n "Preflight: ${DECODER_POD} -> ${PREFILLER_ADDR}:${PREFILLER_HTTP_PORT} ... "
    if oc exec "${DECODER_POD}" -- curl -sf --max-time 5 "http://${PREFILLER_ADDR}:${PREFILLER_HTTP_PORT}/health" > /dev/null 2>&1; then
        echo "ok"
    else
        echo "FAILED" >&2
        echo "ERROR: decoder pod cannot reach prefiller HTTP port. Check NetworkPolicy / routing." >&2
        exit 1
    fi
fi

wait_for_health "${DECODER_POD}"   "Decoder"   "${DECODER_ADDR}"   "${DECODER_HTTP_PORT}"   "${DECODER_LOG}"   || exit 1

# ---------------------------------------------------------------------------
# Start Proxy
# ---------------------------------------------------------------------------
echo "=== Starting Proxy ==="
if [[ "${CONNECTOR}" == "pd_connector" ]]; then
    _DECODER_FIRST_FLAG=""
    [[ "${DECODER_FIRST}" == "true" ]] && _DECODER_FIRST_FLAG="--decoder-first"
    _P2P_FLAG=""
    [[ "${P2P_MODE}" == "true" ]] && _P2P_FLAG="--p2p"
    # The proxy round-robins across each role's DP replicas: prefill gets an
    # X-data-parallel-rank header and the decode remote_port is base + that
    # rank; decode gets its own round-robin rank header. --p2p-connector-port
    # is the prefiller P2P base (proxy adds the per-request rank offset).
    PROXY_PID=$(oc exec "${PROXY_POD}" -- bash -c "
        nohup ${PYTHON_BIN} /tmp/p2p_connector_proxy.py \
            --port ${PROXY_PORT} \
            --host 127.0.0.1 \
            --prefiller-hosts ${PREFILLER_ADDR} \
            --prefiller-ports ${PREFILLER_HTTP_PORT} \
            --decoder-hosts ${DECODER_ADDR} \
            --decoder-ports ${DECODER_HTTP_PORT} \
            --p2p-connector-host ${PREFILLER_ADDR} \
            --p2p-connector-port ${P2P_PORT} \
            --decoder-p2p-connector-host ${DECODER_ADDR} \
            --decoder-p2p-connector-port ${DECODER_P2P_PORT} \
            --prefiller-dp-size ${PREFILLER_DP} \
            --decoder-dp-size ${DECODER_DP} \
            ${_DECODER_FIRST_FLAG} \
            ${_P2P_FLAG} \
            > ${PROXY_LOG} 2>&1 &
        echo \$!
    " 2>/dev/null | tail -1)
else
    PROXY_PID=$(oc exec "${PROXY_POD}" -- bash -c "
        nohup ${PYTHON_BIN} /tmp/toy_proxy_server.py \
            --port ${PROXY_PORT} \
            --host 127.0.0.1 \
            --prefiller-hosts ${PREFILLER_ADDR} \
            --prefiller-ports ${PREFILLER_HTTP_PORT} \
            --decoder-hosts ${DECODER_ADDR} \
            --decoder-ports ${DECODER_HTTP_PORT} \
            > ${PROXY_LOG} 2>&1 &
        echo \$!
    " 2>/dev/null | tail -1)
fi
echo "  PID ${PROXY_PID} → ${PROXY_LOG}"

wait_for_health "${PROXY_POD}" "Proxy" "127.0.0.1" "${PROXY_PORT}" "${PROXY_LOG}" "/healthcheck" || exit 1

# ---------------------------------------------------------------------------
# Write state file (consumed by bench_sweep.sh)
# ---------------------------------------------------------------------------
cat > "${STATE_FILE}" <<STATE
# Generated by deploy.sh — $(date)
CONNECTOR=${CONNECTOR}
DISABLE_OFFLOADING=${DISABLE_OFFLOADING}
P2P_MODE=${P2P_MODE}
PREFILLER_POD=${PREFILLER_POD}
DECODER_POD=${DECODER_POD}
PROXY_POD=${PROXY_POD}
PREFILLER_ADDR=${PREFILLER_ADDR}
DECODER_ADDR=${DECODER_ADDR}
PREFILLER_HTTP_PORT=${PREFILLER_HTTP_PORT}
DECODER_HTTP_PORT=${DECODER_HTTP_PORT}
PROXY_PORT=${PROXY_PORT}
PREFILLER_PID=${PREFILLER_PID}
DECODER_PID=${DECODER_PID}
PROXY_PID=${PROXY_PID}
MODEL=${MODEL}
PROXY_URL=http://127.0.0.1:${PROXY_PORT}
STATE

cat <<SUMMARY

=== Deploy complete: ${CONNECTOR} ===
  Model:    ${MODEL}  gpu_mem=${GPU_MEM_UTIL}  max_len=${MAX_MODEL_LEN}
  Proxy:    http://127.0.0.1:${PROXY_PORT}/v1/completions
  State:    ${STATE_FILE}

  To stop:
    oc exec ${PREFILLER_POD} -- kill ${PREFILLER_PID} ${DECODER_PID} ${PROXY_PID}

  Example request:
    oc exec ${PROXY_POD} -- curl http://127.0.0.1:${PROXY_PORT}/v1/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"${MODEL}","prompt":"Hello","max_tokens":20}'

SUMMARY
