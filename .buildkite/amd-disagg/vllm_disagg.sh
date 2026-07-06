#!/bin/bash
# =============================================================================
# vLLM disaggregated (P/D) launcher (rank-based / SLURM-native).
# -----------------------------------------------------------------------------
#
#   rank 0 .. xP-1        -> prefill   (rank 0 is also the orchestrator + proxy)
#   rank xP .. xP+yD-1    -> decode
#
# Rank 0 additionally: starts the MoRIIO proxy, health-gates every server,
# runs the post-health workload (RUN_AFTER_HEALTH=bench|accuracy|none), then
# drops a shared-FS completion sentinel so the other ranks shut down and the
# single `srun` returns. Rank 0's exit code is the run's pass/fail.
#
# Explicit per-role invocation is also supported (granular for manual use):
# proxy | prefill | decode | bench | accuracy.
#
# ---- LAUNCH IT (both supported) --------------------------------------------
# 1) Single srun (SLURM provides the rank) — what run_xPyD_disagg.slurm does:
#      srun --nodes=$((xP+yD)) --ntasks-per-node=1 \
#           bash .../vllm_disagg.sh node
#    (IPADDRS must be ordered to match ranks: prefill IPs first, then decode,
#     in the SAME order srun assigns PROCID — sort nodes alphabetically.)
#
# 2) Manual, one shell per node (rank from NODE_RANK; share LOG_PATH/FS):
#      IPADDRS=ipP0,ipD0 xP=1 yD=1 NODE_RANK=0 bash vllm_disagg.sh node
#      IPADDRS=ipP0,ipD0 xP=1 yD=1 NODE_RANK=1 bash vllm_disagg.sh node
#
# 3) Manual, fully granular (per-role):
#      bash vllm_disagg.sh proxy
#      bash vllm_disagg.sh prefill
#      NODE_RANK=1 bash vllm_disagg.sh decode
#      bash vllm_disagg.sh bench
#
# Cluster config: cluster.sh (sourced). Model flags: models.yaml.
# Parallelism: WIDE_EP_MODE=0 tp (independent TP servers) | 1 ep (DP+EP groups).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_PROXY_IP_OVERRIDE="${PROXY_IP:-}"

log() { echo "[vllm_disagg] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    echo "Usage: $0 [node|proxy|prefill|decode|bench|accuracy] [--wide-ep-mode 0|1]" >&2
    echo "       (no role => 'node': derive role from \$SLURM_PROCID/\$NODE_RANK)" >&2
    exit 1
}

# ----------------------------------------------------------------- arg parsing
# Sets globals: ROLE (default 'node'), _WIDE_EP_MODE_OVERRIDE.
parse_args() {
    ROLE="${1:-node}"
    shift || true
    _WIDE_EP_MODE_OVERRIDE=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --wide-ep-mode)   _WIDE_EP_MODE_OVERRIDE="${2:-}"; shift 2 ;;
            --wide-ep-mode=*) _WIDE_EP_MODE_OVERRIDE="${1#*=}"; shift ;;
            -h|--help) usage ;;
            *) die "Unknown argument: $1" ;;
        esac
    done
}

# ----------------------------------------------------------------- load config
# Sources cluster.sh and resolves WIDE_EP_MODE -> PARALLEL_MODE (tp|ep).
load_config() {
    CLUSTER_ENV="${CLUSTER_ENV:-${SCRIPT_DIR}/cluster.sh}"
    [[ -f "${CLUSTER_ENV}" ]] || die "cluster env file not found: ${CLUSTER_ENV}"
    # shellcheck disable=SC1090
    source "${CLUSTER_ENV}"

    [[ -n "${_WIDE_EP_MODE_OVERRIDE}" ]] && WIDE_EP_MODE="${_WIDE_EP_MODE_OVERRIDE}"
    WIDE_EP_MODE="${WIDE_EP_MODE:-0}"
    case "${WIDE_EP_MODE}" in
        0) PARALLEL_MODE="tp" ;;
        1) PARALLEL_MODE="ep" ;;
        *) die "WIDE_EP_MODE must be 0 (tp) or 1 (ep); got '${WIDE_EP_MODE}'" ;;
    esac

    # node-mode orchestration knobs
    RUN_AFTER_HEALTH="${RUN_AFTER_HEALTH:-accuracy}"     # bench | accuracy | none
    HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-2400}"

    # MoRIIO KV transfer direction. 0 (default)
    MORIIO_READ_MODE="${MORIIO_READ_MODE:-0}"
}

# Emits the `read_mode` KV-config fragment (leading comma + newline) when read
# mode is enabled, else nothing (leaving MoRIIO's write-mode default).
moriio_read_mode_kv() {
    case "${MORIIO_READ_MODE:-0}" in
        1|true|True|TRUE|yes|on) printf ',\n    "read_mode": true' ;;
        *) : ;;
    esac
}

# ----------------------------------------------------------------- topology
# Resolves MODEL_PATH, MODELS_YAML, the IP array, and the master/proxy addresses.
# IP_ARRAY[0..xP-1] = prefill nodes, [xP..xP+yD-1] = decode nodes.
resolve_topology() {
    MODEL_PATH="${MODEL_DIR%/}/${MODEL_NAME}"
    MODELS_YAML="${MODELS_YAML:-${SCRIPT_DIR}/models.yaml}"
    mkdir -p "${LOG_PATH}" 2>/dev/null || true

    [[ -z "${IPADDRS}" ]] && IPADDRS="${PREFILL_IP},${DECODE_IP}"   # 1P1D fallback
    IFS=',' read -ra IP_ARRAY <<< "${IPADDRS}"
    PREFILL_MASTER_ADDR="${IP_ARRAY[0]}"
    DECODE_MASTER_ADDR="${IP_ARRAY[$xP]:-${IP_ARRAY[-1]}}"
    # Proxy is co-located on the prefill master (rank 0) unless the CALLER
    # explicitly set PROXY_IP. We use the pre-cluster.sh override (captured at the
    # top) — not the live PROXY_IP, which cluster.sh always populates with the
    # static PREFILL_IP fallback and would otherwise mask the real master.
    PROXY_IP="${_PROXY_IP_OVERRIDE:-${PREFILL_MASTER_ADDR}}"
}

# ============================================================================
# Non-server roles
# ============================================================================

run_proxy() {
    [[ -f "${PROXY_SCRIPT}" ]] || die "proxy script not found: ${PROXY_SCRIPT}"
    log "proxy: http=:${PROXY_PORT} discovery=:${PROXY_PING_PORT} (${PROXY_SCRIPT})"
    exec python3 "${PROXY_SCRIPT}" --port "${PROXY_PORT}"
}

# Start the proxy in the BACKGROUND (node-mode rank 0). Sets PROXY_PID.
start_proxy_bg() {
    [[ -f "${PROXY_SCRIPT}" ]] || die "proxy script not found: ${PROXY_SCRIPT}"
    local plog="${LOG_PATH}/proxy_$(date +%Y%m%d_%H%M%S).log"
    log "proxy(bg): http=:${PROXY_PORT} discovery=:${PROXY_PING_PORT} log=${plog}"
    python3 "${PROXY_SCRIPT}" --port "${PROXY_PORT}" >"${plog}" 2>&1 &
    PROXY_PID=$!
}

run_bench() {
    local base_url="http://127.0.0.1:${GATEWAY_PORT:-${PROXY_PORT}}"
    local result_dir="${LOG_PATH}/bench_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${result_dir}"
    log "bench -> ${base_url} model=${MODEL_PATH} wide_ep_mode=${WIDE_EP_MODE}(${PARALLEL_MODE})"
    log "combinations='${BENCHMARK_COMBINATIONS}' concurrency='${BENCHMARK_CON}'"

    local combo isl osl con nump logf
    for combo in ${BENCHMARK_COMBINATIONS}; do
        isl="${combo%%/*}"; osl="${combo##*/}"
        for con in ${BENCHMARK_CON}; do
            nump=$(( NUM_PROMPTS_FACTOR * con ))
            (( nump < BENCHMARK_MIN_PROMPTS )) && nump="${BENCHMARK_MIN_PROMPTS}"
            logf="${result_dir}/bench_${isl}_${osl}_con${con}.log"
            log "ISL=${isl} OSL=${osl} CON=${con} num_prompts=${nump} -> ${logf}"
            vllm bench serve \
                --backend openai \
                --base-url "${base_url}" \
                --endpoint /v1/completions \
                --model "${MODEL_PATH}" \
                --dataset-name random \
                --random-input-len "${isl}" \
                --random-output-len "${osl}" \
                --num-prompts "${nump}" \
                --max-concurrency "${con}" \
                --percentile-metrics ttft,tpot,itl,e2el \
                --ignore-eos \
                2>&1 | tee "${logf}"
        done
    done
    log "bench complete. Results in ${result_dir}"
}

run_accuracy() {
    # lm_eval isn't in the stock vLLM image; install on demand (rank 0 only, and
    # only when accuracy is actually selected). Override with ACCURACY_PIP_SPEC.
    if ! command -v lm_eval >/dev/null 2>&1; then
        log "lm_eval not found — installing ${ACCURACY_PIP_SPEC:-lm_eval[api]}"
        python3 -m pip install --no-cache-dir "${ACCURACY_PIP_SPEC:-lm_eval[api]}"
    fi
    local base_url="http://127.0.0.1:${GATEWAY_PORT:-${PROXY_PORT}}/v1/completions"
    local ts; ts="$(date +%Y%m%d_%H%M%S)"
    local logf="${LOG_PATH}/accuracy_${MODEL_NAME}_${PARALLEL_MODE}_${ts}.log"
    local outdir="${LOG_PATH}/lm_eval_${MODEL_NAME}_${PARALLEL_MODE}_${ts}"
    log "accuracy -> ${base_url} model=${MODEL_PATH} tasks=${ACCURACY_TASKS} wide_ep_mode=${WIDE_EP_MODE}(${PARALLEL_MODE})"
    log "log=${logf} results=${outdir}"

    # --output_path makes lm_eval persist a results_*.json for scraping. 
    local eval_rc=0
    ( set -o pipefail
      # Model weights are already served (loaded offline in a separate process);
      # allow the Hub only here so lm_eval can fetch the eval dataset (e.g. gsm8k).
      export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
      python3 -m lm_eval --model local-completions \
        --tasks "${ACCURACY_TASKS}" \
        --model_args "model=${MODEL_PATH},base_url=${base_url},num_concurrent=${ACCURACY_NUM_CONCURRENT},max_retries=${ACCURACY_MAX_RETRIES},tokenized_requests=False" \
        --output_path "${outdir}" \
        2>&1 | tee "${logf}"
    ) || eval_rc=$?
    if (( eval_rc != 0 )); then
        log "FAIL: lm_eval exited rc=${eval_rc} (no usable score) — Log: ${logf}"
        return "${eval_rc}"
    fi

    # Gate on the score. Find the results JSON lm_eval just wrote, pull the max
    # value of ACCURACY_METRIC across filters/tasks, and compare to the threshold.
    local results_json=""
    results_json="$(find "${outdir}" -name 'results_*.json' -type f 2>/dev/null | sort | tail -n1)"
    if [[ -z "${results_json}" ]]; then
        log "FAIL: no results_*.json under ${outdir} — cannot evaluate threshold"
        return 1
    fi

    local score=""
    score="$(python3 - "${results_json}" "${ACCURACY_METRIC}" <<'PY'
import json, sys
path, metric = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
best = None
for task, metrics in (data.get("results") or {}).items():
    for key, val in metrics.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        if key.split(",")[0] == metric:
            best = val if best is None else max(best, val)
print("" if best is None else "%.6f" % best)
PY
)" || score=""

    if [[ -z "${score}" ]]; then
        log "FAIL: could not parse metric '${ACCURACY_METRIC}' from ${results_json}"
        return 1
    fi

    if awk -v s="${score}" -v t="${ACCURACY_THRESHOLD}" 'BEGIN{exit !(s+0 >= t+0)}'; then
        log "PASS: ${ACCURACY_METRIC}=${score} >= threshold=${ACCURACY_THRESHOLD} (tasks=${ACCURACY_TASKS}). Log: ${logf}"
        return 0
    fi
    log "FAIL: ${ACCURACY_METRIC}=${score} < threshold=${ACCURACY_THRESHOLD} (tasks=${ACCURACY_TASKS}). Log: ${logf}"
    return 1
}

# ============================================================================
# Server roles (prefill / decode) and their shared helpers
# ============================================================================

# Apply vLLM PR #39276 when required (WIDE_EP_MODE=1 and xP>1 or yD>1), honoring
# APPLY_MORIIO_PATCH=auto|1|0. Aborts if a required patch is missing/fails.
# TODO: Remove this logic after PR #45043 is merged. 
apply_patch_if_needed() {
    local patch_required=0
    if [[ "${WIDE_EP_MODE}" == "1" ]] && { [[ "${xP}" -gt 1 ]] || [[ "${yD}" -gt 1 ]]; }; then
        patch_required=1
    fi
    local patch_script="${PATCH_SCRIPT:-${SCRIPT_DIR}/apply_moriio_2pd_patches.sh}"
    local do_patch=0
    case "${APPLY_MORIIO_PATCH}" in
        1) do_patch=1 ;;
        0) do_patch=0 ;;
        auto|*) do_patch="${patch_required}" ;;
    esac
    [[ "${do_patch}" == "1" ]] || return 0

    if [[ -f "${patch_script}" ]]; then
        log "applying MoRIIO multi-node patch (PR #39276): ${patch_script}"
        if ! bash "${patch_script}"; then
            [[ "${patch_required}" == "1" ]] && die "patch required for multi-node DP (xP=${xP} yD=${yD}) but failed"
            log "WARN: patch failed; continuing (not strictly required for 1P1D)"
        fi
    elif [[ "${patch_required}" == "1" ]]; then
        die "patch script not found but required for multi-node DP: ${patch_script}"
    fi
}

# Read model-specific flags from models.yaml (mode + role aware) into globals:
#   MODEL_BASE_FLAGS / MODEL_ROLE_FLAGS / MODEL_EXPERIMENTAL_FLAGS -> MODEL_CONFIG
# Also exports the model's env: block (only if not already set; caller env wins).
load_model_flags() {
    [[ -f "${MODELS_YAML}" ]] || die "models.yaml not found: ${MODELS_YAML}"
    export MODELS_YAML MODEL_NAME PARALLEL_MODE ROLE
    eval "$(python3 - <<'PY'
import os, shlex, sys, yaml
path = os.environ["MODELS_YAML"]; name = os.environ["MODEL_NAME"]
mode = os.environ["PARALLEL_MODE"]; role = os.environ["ROLE"]
with open(path, "r", encoding="utf-8") as f:
    doc = yaml.safe_load(f) or {}
# Preferred form: top-level `models:` list of {model: <name>, ...} entries.
# Back-compat: a top-level mapping keyed by model name.
entries = doc.get("models") if isinstance(doc, dict) else doc
cfg = None
if isinstance(entries, list):
    cfg = next((e for e in entries if isinstance(e, dict) and e.get("model") == name), None)
elif isinstance(doc, dict):
    cfg = doc.get(name)
if cfg is None:
    print(f'echo "ERROR: model {name} not found in {path}" >&2; exit 1'); sys.exit(0)
role_cfg = cfg.get(role, {}) or {}
def q(v): return shlex.quote(str(v if v is not None else ""))
exports = {
    "MODEL_BASE_FLAGS": cfg.get("base_flags", "") or "",
    "MODEL_ROLE_FLAGS": role_cfg.get(mode, "") or "",
    "MODEL_EXPERIMENTAL_FLAGS": cfg.get("experimental_flags", "") or "",
}
for k, v in exports.items(): print(f"{k}={q(v)}")
# Model-specific env: exported only if not already set (caller env wins).
for k, v in (cfg.get("env", {}) or {}).items():
    if k not in os.environ:
        print(f"export {k}={q(v)}")
PY
)"
    MODEL_CONFIG="${MODEL_BASE_FLAGS} ${MODEL_ROLE_FLAGS} ${MODEL_EXPERIMENTAL_FLAGS}"
}

# Build the TP-mode (WIDE_EP_MODE=0) serve command into the global CMD array.
# Each node is an independent TP server registered to the proxy.
build_tp_cmd() {
    local port="${TP_PORT}"
    [[ -n "${HOST_IP}" ]] && CMD+=(--host "${HOST_IP}")
    CMD+=(--port "${port}" --tensor-parallel-size "${TP_SIZE}")
    local _mc; read -ra _mc <<< "${MODEL_CONFIG}"; CMD+=("${_mc[@]}")
    local kv_cfg read_mode_kv
    read_mode_kv="$(moriio_read_mode_kv)"
    kv_cfg=$(cat <<EOF
{
  "kv_connector": "MoRIIOConnector",
  "kv_role": "${KV_ROLE}",
  "kv_connector_extra_config": {
    "proxy_ip": "${PROXY_IP}",
    "proxy_ping_port": "${PROXY_PING_PORT}",
    "http_port": "${port}",
    "handshake_port": "${HANDSHAKE_PORT}",
    "notify_port": "${NOTIFY_PORT}"${read_mode_kv}
  }
}
EOF
)
    CMD+=(--kv-transfer-config "${kv_cfg}")
    log "role=${ROLE} mode=tp host=${HOST_IP:-0.0.0.0}:${port} kv_role=${KV_ROLE} tp=${TP_SIZE} read_mode=${MORIIO_READ_MODE}"
}

# Build the EP-mode (WIDE_EP_MODE=1) serve command into the global CMD array.
# DP + expert parallel; masters expose the API server + KV transfer, children
# join headless.
build_ep_cmd() {
    #   prefill -> mori_high_throughput  (paired with --enforce-eager in models.yaml)
    #   decode  -> mori_low_latency      (paired with CUDA graphs in models.yaml)
    local default_backend="mori_low_latency"
    [[ "${ROLE}" == "prefill" ]] && default_backend="mori_high_throughput"
    local backend="${ALL2ALL_BACKEND:-${default_backend}}"
    export VLLM_ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-${backend}}"
    local port="${SERVE_PORT}"
    CMD+=(
        -tp 1
        --data-parallel-size "${DP_GROUP_SIZE}"
        --data-parallel-size-local "${GPUS_PER_NODE}"
        --data-parallel-address "${DP_MASTER_ADDR}"
        --data-parallel-rpc-port "${RPC_PORT}"
        --enable-expert-parallel
        --all2all-backend "${backend}"
        --port "${port}"
        --no-enable-prefix-caching
        --distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS}"
    )
    local _mc; read -ra _mc <<< "${MODEL_CONFIG}"; CMD+=("${_mc[@]}")

    if [[ "${IS_MASTER}" == "1" ]]; then
        CMD+=(--api-server-count="${GPUS_PER_NODE}")
        local kv_cfg read_mode_kv
        read_mode_kv="$(moriio_read_mode_kv)"
        kv_cfg=$(cat <<EOF
{
  "kv_connector": "MoRIIOConnector",
  "kv_role": "${KV_ROLE}",
  "kv_port": "${KV_PORT}",
  "kv_connector_extra_config": {
    "proxy_ip": "${PROXY_IP}",
    "proxy_port": "${PROXY_PORT}",
    "proxy_ping_port": "${PROXY_PING_PORT}",
    "http_port": "${port}",
    "local_ping_port": "${LOCAL_PING_PORT}",
    "handshake_port": "${HANDSHAKE_PORT}",
    "notify_port": "${NOTIFY_PORT}"${read_mode_kv}
  }
}
EOF
)
        CMD+=(--kv-transfer-config "${kv_cfg}")
        log "role=${ROLE} mode=ep MASTER node_rank=${NODE_RANK} dp_size=${DP_GROUP_SIZE} dp_addr=${DP_MASTER_ADDR} kv_role=${KV_ROLE} all2all=${backend} read_mode=${MORIIO_READ_MODE}"
    else
        CMD+=(--data-parallel-start-rank "${DP_START_RANK}" --headless)
        log "role=${ROLE} mode=ep CHILD  node_rank=${NODE_RANK} dp_size=${DP_GROUP_SIZE} dp_start_rank=${DP_START_RANK} dp_addr=${DP_MASTER_ADDR} all2all=${backend}"
    fi
}

# Set the prefill topology globals for the current NODE_RANK (no execution).
configure_prefill() {
    : "${NODE_RANK:=0}"                    # manual default: prefill master
    KV_ROLE="kv_producer"
    DP_GROUP_SIZE=$(( xP * GPUS_PER_NODE ))
    DP_MASTER_ADDR="${PREFILL_MASTER_ADDR}"
    DP_START_RANK=$(( NODE_RANK * GPUS_PER_NODE ))
    [[ "${NODE_RANK}" -eq 0 ]] && IS_MASTER=1 || IS_MASTER=0
    TP_PORT="${PREFILL_PORT}"
}

# Set the decode topology globals for the current NODE_RANK (no execution).
configure_decode() {
    : "${NODE_RANK:=${xP}}"               # manual default: decode master
    KV_ROLE="kv_consumer"
    DP_GROUP_SIZE=$(( yD * GPUS_PER_NODE ))
    DP_MASTER_ADDR="${DECODE_MASTER_ADDR}"
    DP_START_RANK=$(( (NODE_RANK - xP) * GPUS_PER_NODE ))
    [[ "${NODE_RANK}" -eq "${xP}" ]] && IS_MASTER=1 || IS_MASTER=0
    TP_PORT="${DECODE_PORT}"
}

# Build the serve command (patch + model flags + CMD array + LOGF). No exec.
# Expects role globals set by configure_prefill/configure_decode.
build_server_cmd() {
    HOST_IP="${IP_ARRAY[$NODE_RANK]:-}"   # own IP for --host binding, if known
    apply_patch_if_needed
    load_model_flags
    LOGF="${LOG_PATH}/${ROLE}_${PARALLEL_MODE}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
    CMD=(vllm serve "${MODEL_PATH}")
    if [[ "${WIDE_EP_MODE}" == "0" ]]; then build_tp_cmd; else build_ep_cmd; fi
    log "model=${MODEL_PATH}"
}

# Foreground server run (explicit prefill/decode roles — original behavior).
run_server_fg() {
    build_server_cmd
    log "log=${LOGF}"
    "${CMD[@]}" 2>&1 | tee "${LOGF}"
}

run_prefill() { configure_prefill; run_server_fg; }
run_decode()  { configure_decode;  run_server_fg; }

# ============================================================================
# node mode — SLURM-native rank-based self-select + rank-0 orchestration
# ============================================================================

# Endpoints to health-check, one per line as "ip:port", depending on mode.
#   tp: every prefill node (:PREFILL_PORT) and every decode node (:DECODE_PORT)
#   ep: only the masters expose an API (prefill rank 0, decode rank xP) :SERVE_PORT
health_endpoints() {
    local i
    if [[ "${WIDE_EP_MODE}" == "0" ]]; then
        for (( i=0; i<xP; i++ ));        do echo "${IP_ARRAY[$i]}:${PREFILL_PORT}"; done
        for (( i=xP; i<xP+yD; i++ ));     do echo "${IP_ARRAY[$i]}:${DECODE_PORT}";  done
    else
        echo "${IP_ARRAY[0]}:${SERVE_PORT}"
        echo "${IP_ARRAY[$xP]}:${SERVE_PORT}"
    fi
}

# Block until every endpoint's /health is OK, or timeout, or our local server dies.
wait_all_healthy() {
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
    local eps; mapfile -t eps < <(health_endpoints)
    log "health-gate: waiting on ${#eps[@]} endpoint(s): ${eps[*]}"
    local ep
    for ep in "${eps[@]}"; do
        until curl -sf "http://${ep}/health" >/dev/null 2>&1; do
            (( $(date +%s) >= deadline )) && { log "TIMEOUT waiting for ${ep}"; return 1; }
            kill -0 "${SERVER_PID}" 2>/dev/null || { log "local server (pid ${SERVER_PID}) exited while waiting"; return 1; }
            sleep 10
        done
        log "healthy: ${ep}"
    done
    return 0
}

run_workload() {
    case "${RUN_AFTER_HEALTH}" in
        bench)    run_bench ;;
        accuracy) run_accuracy ;;
        none|"")  log "RUN_AFTER_HEALTH=none — skipping workload" ;;
        *)        die "unknown RUN_AFTER_HEALTH='${RUN_AFTER_HEALTH}'" ;;
    esac
}

# rank 0: proxy + health-gate + workload, then write the completion sentinel.
orchestrate_master() {
    local sentinel="$1" rc=0
    # Front door: the toy proxy runs in-container (started here); the vllm-router
    # runs as a SEPARATE container started by the SLURM job on this (rank-0) node,
    # so in that mode we don't start anything here.
    PROXY_PID=""
    if [[ "${ROUTER_TYPE:-toy}" == "vllm-router" ]]; then
        log "ROUTER_TYPE=vllm-router: external router expected on gateway :${GATEWAY_PORT:-${ROUTER_PORT}} (not starting toy proxy)"
    else
        start_proxy_bg
    fi
    if wait_all_healthy; then
        set +e
        run_workload
        rc=$?
        set -e
    else
        rc=1
    fi
    echo "${rc}" > "${sentinel}" 2>/dev/null || true
    log "master: workload rc=${rc}; sentinel=${sentinel}; tearing down local proxy+server"
    [[ -n "${PROXY_PID:-}" ]] && kill "${PROXY_PID}" 2>/dev/null || true
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    return "${rc}"
}

# non-master ranks: keep serving until the master signals completion (sentinel)
# or our local server dies unexpectedly.
watch_until_done() {
    local sentinel="$1" rc=0
    log "rank ${NODE_RANK} serving; waiting for completion sentinel (${sentinel})"
    while :; do
        [[ -f "${sentinel}" ]] && { log "completion sentinel seen; shutting down"; break; }
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "local server (pid ${SERVER_PID}) exited before completion"; rc=1; break
        fi
        sleep 10
    done
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    return "${rc}"
}

run_node() {
    NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
    local total=$(( xP + yD ))
    (( NODE_RANK >= 0 && NODE_RANK < total )) \
        || die "NODE_RANK=${NODE_RANK} out of range [0,${total}) (xP=${xP} yD=${yD})"

    if (( NODE_RANK < xP )); then ROLE="prefill"; configure_prefill
    else                          ROLE="decode";  configure_decode
    fi

    # Shared-FS completion sentinel (LOG_PATH is shared & per-run/per-job).
    local sentinel="${LOG_PATH}/.disagg_done"
    (( NODE_RANK == 0 )) && { rm -f "${sentinel}" 2>/dev/null || true; }

    log "node mode: rank=${NODE_RANK}/${total} role=${ROLE} master=${IS_MASTER} wide_ep=${WIDE_EP_MODE}(${PARALLEL_MODE})"
    log "topology: xP=${xP} yD=${yD} gpus/node=${GPUS_PER_NODE} IPADDRS=${IPADDRS}"

    # DRY_RUN: resolve+print the plan (role, serve cmd, endpoints) and exit 0.
    # Validates wiring without launching vLLM / the proxy / any workload.
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        build_server_cmd
        log "DRY_RUN: server cmd: ${CMD[*]}"
        if (( NODE_RANK == 0 )); then
            local eps; mapfile -t eps < <(health_endpoints)
            log "DRY_RUN: health endpoints: ${eps[*]}"
            log "DRY_RUN: post-health workload: ${RUN_AFTER_HEALTH}"
            log "DRY_RUN: sentinel: ${sentinel}"
        fi
        log "DRY_RUN complete (rank ${NODE_RANK}) — no processes launched"
        exit 0
    fi

    # aiter and flydsl keep runtime JIT/kernel caches inside the aiter install
    # tree (…/aiter/jit and …/aiter/jit/flydsl_cache), which is read-only for the
    # non-root (--user) container. AITER_JIT_DIR / FLYDSL_RUNTIME_CACHE_DIR
    # redirect those to node-local /tmp (both are sanctioned overrides: aiter only
    # points flydsl at its bundled cache when FLYDSL_RUNTIME_CACHE_DIR is unset).
    # Seed each once from the image's bundled copy so we keep the prebuilt
    # modules/kernels (fast, no NFS, no from-source rebuild / MLIR recompile).
    local _aiter_src
    _aiter_src="$(python3 -c 'import os,aiter; print(os.path.join(os.path.dirname(aiter.__file__),"jit"))' 2>/dev/null || true)"
    if [[ -n "${AITER_JIT_DIR:-}" && ! -e "${AITER_JIT_DIR}/.seeded" ]]; then
        if [[ -n "${_aiter_src}" && -d "${_aiter_src}" ]]; then
            mkdir -p "${AITER_JIT_DIR}"
            cp -a "${_aiter_src}/." "${AITER_JIT_DIR}/" 2>/dev/null || true
            touch "${AITER_JIT_DIR}/.seeded" 2>/dev/null || true
            log "seeded AITER_JIT_DIR=${AITER_JIT_DIR} from ${_aiter_src}"
        else
            log "WARN: could not locate aiter jit dir to seed AITER_JIT_DIR=${AITER_JIT_DIR}; aiter may rebuild from source"
        fi
    fi
    if [[ -n "${FLYDSL_RUNTIME_CACHE_DIR:-}" && ! -e "${FLYDSL_RUNTIME_CACHE_DIR}/.seeded" ]]; then
        mkdir -p "${FLYDSL_RUNTIME_CACHE_DIR}"
        if [[ -n "${_aiter_src}" && -d "${_aiter_src}/flydsl_cache" ]]; then
            cp -a "${_aiter_src}/flydsl_cache/." "${FLYDSL_RUNTIME_CACHE_DIR}/" 2>/dev/null || true
            log "seeded FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR} from ${_aiter_src}/flydsl_cache"
        else
            log "FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR} set; no bundled flydsl_cache to seed (will JIT-compile at runtime)"
        fi
        touch "${FLYDSL_RUNTIME_CACHE_DIR}/.seeded" 2>/dev/null || true
    fi

    # mori JIT-compiles its shmem/all2all kernels to ~/.mori/jit/<arch>_<nic>/ on
    # first EP use. HOME here is /workspace, bind-mounted to persistent+shared NFS
    # (/data/$USER), so a wrong-arch cache from an earlier run/build survives and
    # gets reloaded even after MORI_GPU_ARCHS changes -- an existing cache dir
    # wins over the arch setting. Reloading a gfx942 shmem_kernels.hsaco on gfx950
    # hardware fails with "device kernel image is invalid" and every EP worker
    # dies at init (surfacing async as a torch.tensor HIP error). Guard: on EP
    # runs, if the cache has no build for our target arch, purge it so mori
    # recompiles for MORI_GPU_ARCHS. A matching arch dir is kept (fast reload).
    if [[ "${WIDE_EP_MODE}" == "1" && -n "${MORI_GPU_ARCHS:-}" ]]; then
        local _mori_jit="${HOME:-/workspace}/.mori/jit"
        if [[ -d "${_mori_jit}" ]] && ! compgen -G "${_mori_jit}/${MORI_GPU_ARCHS}_*" >/dev/null 2>&1; then
            log "purging stale mori jit cache ${_mori_jit} (no ${MORI_GPU_ARCHS}_* build present; forcing recompile for ${MORI_GPU_ARCHS})"
            rm -rf "${_mori_jit}" 2>/dev/null || true
        fi
    fi

    # Start this node's server in the background (full server log -> LOGF; this
    # shell's stdout stays free for orchestration logs the srun/CI captures).
    build_server_cmd
    "${CMD[@]}" >"${LOGF}" 2>&1 &
    SERVER_PID=$!
    log "server started pid=${SERVER_PID} log=${LOGF}"

    local rc=0
    if (( NODE_RANK == 0 )); then
        orchestrate_master "${sentinel}" || rc=$?
    else
        watch_until_done "${sentinel}" || rc=$?
    fi
    log "node rank=${NODE_RANK} exiting rc=${rc}"
    exit "${rc}"
}

# ============================================================================
main() {
    parse_args "$@"
    load_config
    resolve_topology
    case "${ROLE}" in
        node)     run_node ;;
        proxy)    run_proxy ;;
        bench)    run_bench ;;
        accuracy) run_accuracy ;;
        prefill)  run_prefill ;;
        decode)   run_decode ;;
        *)        usage ;;
    esac
}

main "$@"
