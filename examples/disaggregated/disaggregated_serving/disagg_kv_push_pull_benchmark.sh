#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# P/D Disaggregation Benchmark (pull / push modes)
#
# Launches the appropriate disaggregated-serving proxy from your local vLLM
# clone, drives it with `vllm bench serve`, and prints a short summary.
#
#   pull  - decode pulls KV from prefill   (disagg_proxy_demo.py)
#   push  - prefill pushes KV to decode    (disagg_proxy_pushconnector_demo.py)
#
# Prerequisites:
#   - A prefill (P) and a decode (D) vLLM server already running.
#   - A local vLLM checkout (set VLLM_SRC to point at it).
#   - Python deps for the proxy: pip install aiohttp fastapi uvicorn
#
# Usage:
#   VLLM_SRC=/path/to/vllm \
#   MODEL=/path/to/model \
#   PREFILL_URL=http://<p-host>:8100 DECODE_URL=http://<d-host>:8200 \
#   MODES="pull push" \
#   ./benchmark_kv_push.sh
#
# Common options (environment variables):
#   VLLM_SRC        - Path to your local vLLM clone (required)
#   MODEL           - Model path/name for the tokenizer (required)
#   SERVED_MODEL_NAME - Name the servers were started with (--served-model-name).
#                       Sent in each request; defaults to MODEL if unset.
#   PREFILL_URL     - Prefill server URL   (default: http://localhost:8100)
#   DECODE_URL      - Decode server URL    (default: http://localhost:8200)
#   MODES           - Space-separated modes: pull, push (default: "pull")
#   QPS_LIST        - Space-separated request rates (default: "1 2 4")
#   INPUT_LEN       - Sonnet input length  (default: 2048)
#   OUTPUT_LEN      - Sonnet output length (default: 128)
#   INPUT_LENS      - Space-separated input lengths to sweep  (default: INPUT_LEN)
#   OUTPUT_LENS     - Space-separated output lengths to sweep (default: OUTPUT_LEN)
#   NUM_REQS        - Requests per run     (default: 100)
#   ITERATIONS      - Repeats per (mode,qps) (default: 1)
#   RESULTS_DIR     - Output directory     (default: ./results)
#   DATASET_PATH    - Sonnet dataset file  (default: built from VLLM_SRC)
#
# Push-mode only (must match the prefill server's --kv-transfer-config):
#   PREFILL_ENGINE_ID          - engine_id of the prefill instance (required)
#   PREFILL_KV_HOST            - NIXL side-channel host (default: prefill host)
#   PREFILL_SIDE_CHANNEL_PORT  - NIXL side-channel port (default: 5600)
#   PREFILL_TP_SIZE            - prefill tensor-parallel size (default: 1)
#   PREFILL_PP_SIZE            - prefill pipeline-parallel size (default: 1)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

VLLM_SRC="${VLLM_SRC:-}"
MODEL="${MODEL:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

PREFILL_URL="${PREFILL_URL:-http://localhost:8100}"
DECODE_URL="${DECODE_URL:-http://localhost:8200}"

MODES="${MODES:-pull}"
QPS_LIST="${QPS_LIST:-1 2 4}"
# INPUT_LENS / OUTPUT_LENS accept space-separated lists to sweep multiple
# lengths. The singular INPUT_LEN / OUTPUT_LEN still work as defaults.
INPUT_LENS="${INPUT_LENS:-${INPUT_LEN:-2048}}"
OUTPUT_LENS="${OUTPUT_LENS:-${OUTPUT_LEN:-128}}"
NUM_REQS="${NUM_REQS:-100}"
ITERATIONS="${ITERATIONS:-1}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
PROXY_PORT="${PROXY_PORT:-8000}"

# Push-mode coordinates.
PREFILL_ENGINE_ID="${PREFILL_ENGINE_ID:-}"
PREFILL_KV_HOST="${PREFILL_KV_HOST:-}"
PREFILL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNEL_PORT:-5600}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
PREFILL_PP_SIZE="${PREFILL_PP_SIZE:-1}"

# Proxy scripts from the local vLLM clone.
PROXY_DIR="${VLLM_SRC}/examples/disaggregated/disaggregated_serving"
PULL_PROXY_SCRIPT="${PULL_PROXY_SCRIPT:-${PROXY_DIR}/disagg_proxy_demo.py}"
PUSH_PROXY_SCRIPT="${PUSH_PROXY_SCRIPT:-${PROXY_DIR}/disagg_proxy_pushconnector_demo.py}"

# Sonnet dataset.
DATASET_NAME="sonnet"
DATASET_PATH="${DATASET_PATH:-}"
PREFIX_LEN=50

# ── Functions ────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

kill_proxy() {
    if [ -n "${PROXY_PID:-}" ]; then
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
        unset PROXY_PID
    fi
    lsof -t -i:"$PROXY_PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 1
}

wait_for_server() {
    local url=$1
    local name=$2
    local max_wait=${3:-60}
    log "Waiting for $name at $url ..."
    for _ in $(seq 1 "$max_wait"); do
        if curl -sf "${url}/health" > /dev/null 2>&1 || \
           curl -sf "${url}/v1/models" > /dev/null 2>&1 || \
           curl -sf "${url}/status" > /dev/null 2>&1; then
            log "$name is ready"
            return 0
        fi
        sleep 1
    done
    log "ERROR: $name did not start within ${max_wait}s"
    return 1
}

# Build a 4x-concatenated sonnet file so longer INPUT_LEN values have enough
# source text. Uses the sonnet.txt shipped in the vLLM clone.
ensure_sonnet_dataset() {
    if [ -n "$DATASET_PATH" ] && [ -f "$DATASET_PATH" ]; then
        return
    fi
    local src="${VLLM_SRC}/benchmarks/sonnet.txt"
    if [ ! -f "$src" ]; then
        log "ERROR: sonnet.txt not found at $src. Set DATASET_PATH to a sonnet file."
        exit 1
    fi
    DATASET_PATH="${RESULTS_DIR}/sonnet_4x.txt"
    : > "$DATASET_PATH"
    for _ in 1 2 3 4; do
        cat "$src" >> "$DATASET_PATH"
    done
    log "Created $DATASET_PATH from $src"
}

# Resolve the model name to send in requests (must match the server's
# --served-model-name). Falls back to MODEL when SERVED_MODEL_NAME is unset.
proxy_model_name() {
    if [ -n "$SERVED_MODEL_NAME" ]; then
        echo "$SERVED_MODEL_NAME"
    else
        echo "$MODEL"
    fi
}

# host:port from a URL (both proxies want host:port, not full URLs).
url_to_hostport() {
    local hp="${1#*://}"
    echo "${hp%%/*}"
}

start_pull_proxy() {
    local mode=$1
    local proxy_args=(
        --model "$(proxy_model_name)"
        --prefill "$(url_to_hostport "$PREFILL_URL")"
        --decode "$(url_to_hostport "$DECODE_URL")"
        --port "$PROXY_PORT"
    )
    log "Starting pull proxy: prefill=$(url_to_hostport "$PREFILL_URL") decode=$(url_to_hostport "$DECODE_URL")"
    python3 "$PULL_PROXY_SCRIPT" "${proxy_args[@]}" &
    PROXY_PID=$!
}

start_push_proxy() {
    local mode=$1
    if [ -z "$PREFILL_ENGINE_ID" ]; then
        log "ERROR: push mode requires PREFILL_ENGINE_ID (must match the prefill server's --kv-transfer-config engine_id)"
        return 1
    fi
    local prefill_hostport
    prefill_hostport="$(url_to_hostport "$PREFILL_URL")"
    local prefill_kv_host="$PREFILL_KV_HOST"
    if [ -z "$prefill_kv_host" ]; then
        prefill_kv_host="${prefill_hostport%%:*}"
    fi
    local proxy_args=(
        --model "$(proxy_model_name)"
        --prefill "$prefill_hostport"
        --decode "$(url_to_hostport "$DECODE_URL")"
        --port "$PROXY_PORT"
        --prefill-engine-id "$PREFILL_ENGINE_ID"
        --prefill-kv-host "$prefill_kv_host"
        --prefill-side-channel-port "$PREFILL_SIDE_CHANNEL_PORT"
        --prefill-tp-size "$PREFILL_TP_SIZE"
        --prefill-pp-size "$PREFILL_PP_SIZE"
    )
    log "Starting push proxy: prefill=$prefill_hostport decode=$(url_to_hostport "$DECODE_URL") kv_host=$prefill_kv_host"
    python3 "$PUSH_PROXY_SCRIPT" "${proxy_args[@]}" &
    PROXY_PID=$!
}

start_proxy() {
    local mode=$1
    kill_proxy
    case "$mode" in
        push) start_push_proxy "$mode" || return 1 ;;
        pull) start_pull_proxy "$mode" ;;
        *)    log "ERROR: unknown mode '$mode' (expected pull or push)"; return 1 ;;
    esac
    sleep 2
    wait_for_server "http://localhost:${PROXY_PORT}" "proxy ($mode)" 30
}

run_benchmark() {
    local mode=$1
    local qps=$2
    local iter=$3
    local input_len=$4
    local output_len=$5
    local tag="${mode}_qps${qps}_in${input_len}_out${output_len}_iter${iter}"

    log "Benchmark: mode=$mode qps=$qps input=$input_len output=$output_len reqs=$NUM_REQS iter=$iter/$ITERATIONS"

    local bench_args=(
        --backend vllm
        --model "$MODEL"
        --dataset-name "$DATASET_NAME"
        --dataset-path "$DATASET_PATH"
        --sonnet-input-len "$input_len"
        --sonnet-output-len "$output_len"
        --sonnet-prefix-len "$PREFIX_LEN"
        --num-prompts "$NUM_REQS"
        --port "$PROXY_PORT"
        --save-result
        --save-detailed
        --result-dir "$RESULTS_DIR"
        --result-filename "${tag}.json"
        --request-rate "$qps"
        --temperature 0
        --top-p 1
        --extra-body '{"temperature": 0, "top_p": 1, "seed": 0}'
    )
    if [ -n "$SERVED_MODEL_NAME" ]; then
        bench_args+=(--served-model-name "$SERVED_MODEL_NAME")
    fi

    vllm bench serve "${bench_args[@]}" 2>&1 | tee "${RESULTS_DIR}/${tag}.log"

    log "Completed: $tag"
    sleep 2
}

print_summary() {
    echo ""
    log "═══════════════════════════════════════════════════════════"
    log "  BENCHMARK COMPLETE"
    log "  Results in: $RESULTS_DIR"
    log "═══════════════════════════════════════════════════════════"
    ls -1 "${RESULTS_DIR}"/*.json 2>/dev/null || echo "  (no result files)"
}

# ── Main ─────────────────────────────────────────────────────────────────────

if [ -z "$VLLM_SRC" ]; then
    log "ERROR: VLLM_SRC is not set. Point it at your local vLLM clone."
    exit 1
fi
if [ -z "$MODEL" ]; then
    log "ERROR: MODEL is not set."
    exit 1
fi

trap kill_proxy EXIT

mkdir -p "$RESULTS_DIR"
ensure_sonnet_dataset

for mode in $MODES; do
    proxy_script="$PULL_PROXY_SCRIPT"
    [ "$mode" = "push" ] && proxy_script="$PUSH_PROXY_SCRIPT"
    if [ ! -f "$proxy_script" ]; then
        log "ERROR: proxy script for mode '$mode' not found: $proxy_script"
        exit 1
    fi
done

log "Checking backend servers..."
wait_for_server "$PREFILL_URL" "prefill" 10 || {
    log "Prefill server not reachable at $PREFILL_URL. Start P/D servers first."
    exit 1
}
wait_for_server "$DECODE_URL" "decode" 10 || {
    log "Decode server not reachable at $DECODE_URL. Start P/D servers first."
    exit 1
}

log "Configuration:"
log "  vLLM src:   $VLLM_SRC"
log "  Model:      $MODEL"
log "  Served as:  ${SERVED_MODEL_NAME:-<MODEL>}"
log "  Prefill:    $PREFILL_URL"
log "  Decode:     $DECODE_URL"
log "  Modes:      $MODES"
log "  QPS:        $QPS_LIST"
log "  Input len:  $INPUT_LENS"
log "  Output len: $OUTPUT_LENS"
log "  Requests:   $NUM_REQS"
log "  Iterations: $ITERATIONS"
echo ""

for mode in $MODES; do
    log "━━━ Mode: $mode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    start_proxy "$mode"
    for in_len in $INPUT_LENS; do
        for out_len in $OUTPUT_LENS; do
            for qps in $QPS_LIST; do
                for iter in $(seq 1 "$ITERATIONS"); do
                    run_benchmark "$mode" "$qps" "$iter" "$in_len" "$out_len"
                done
            done
        done
    done
    kill_proxy
done

print_summary
