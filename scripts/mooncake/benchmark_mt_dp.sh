#!/usr/bin/env bash
# Benchmark multi-turn chat with/without CPU KV cache offloading.
# Launches a vllm server, runs `vllm-bench` with multi-turn options,
# then repeats with offloading enabled.
#
# Usage:
#   bash benchmark_mt_dp.sh [MODEL] [INPUT_LEN] [OUTPUT_LEN] [NUM_PROMPTS]
#
# Supported backends (comma-separated in BACKENDS):
#   baseline        - No offloading
#   native          - Built-in vLLM KV offloading (--kv-offloading-backend native)
#   simple          - Simple native offload (VLLM_USE_SIMPLE_KV_OFFLOAD=1 + native backend)
#   mooncake-mem    - MooncakeStoreConnector with CPU memory only (no disk)
#   mooncake        - MooncakeStoreConnector via --kv-transfer-config
#
# Environment variables:
#   CPU_OFFLOAD_GIB       - CPU offload buffer in GiB   (default: 80)
#   DISK_OFFLOAD_GIB      - Disk offload quota in GiB   (default: 400)
#   PORT                  - Server port                  (default: 8192)
#   RESULT_DIR            - Output directory             (default: ./bench_results)
#   BACKENDS              - Comma-separated backends     (default: baseline,mooncake)
#   MOONCAKE_CONFIG_PATH  - Path to mooncake config JSON (required for mooncake, auto-skipped if unset)
#   MULTI_TURN_NUM_TURNS  - Number of turns per conversation  (default: 4)
#   MULTI_TURN_CONCURRENCY - Concurrent conversations         (default: 8)
#   MULTI_TURN_DELAY_MS   - Delay between turns in ms         (default: 500)
#   GLOBAL_PREFIX_RATIO   - Fraction of input as global prefix  (default: 0.1)
#   CONV_PREFIX_RATIO     - Fraction of input as conversation prefix (default: 0.8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
# MODEL="${1:-Qwen/Qwen3-30B-A3B}"
INPUT_LEN="${2:-70000}"
OUTPUT_LEN="${3:-200}"
NUM_PROMPTS="${4:-70}"
CPU_OFFLOAD_GIB="${CPU_OFFLOAD_GIB:-300}"
DISK_OFFLOAD_GIB="${DISK_OFFLOAD_GIB:-2000}"
PORT="${PORT:-8192}"
RESULT_DIR="${RESULT_DIR:-./bench_results}"
BACKENDS="${BACKENDS:-baseline,mooncake}"

MULTI_TURN_NUM_TURNS="${MULTI_TURN_NUM_TURNS:-3}"
MULTI_TURN_CONCURRENCY="${MULTI_TURN_CONCURRENCY:-16}"
MULTI_TURN_DELAY_MS="${MULTI_TURN_DELAY_MS:-30000}"
GLOBAL_PREFIX_RATIO="${GLOBAL_PREFIX_RATIO:-0.1}"
CONV_PREFIX_RATIO="${CONV_PREFIX_RATIO:-0.8}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$RESULT_DIR"

SERVER_COMMON=(
    --model "$MODEL"
    # -tp 4
    -dp 2
    --disable-hybrid-kv-cache-manager
    --port "$PORT"
    --gpu-memory-utilization 0.5
    --no-enable-log-requests
    --load-format dummy
    --attention-backend auto
)
if [[ "$MODEL" == *"Qwen3.5"* ]]; then
    SERVER_COMMON+=(
        --mamba-cache-mode align
        --enable-prefix-caching
    )
fi

if [[ "$MODEL" == "nvidia/Qwen3.5-397B-A17B-NVFP4" ]]; then
    SERVER_COMMON+=(
        -dp 4
        --enable-expert-parallel
        --language-model-only
        --reasoning-parser qwen3
    )
fi

BENCH_COMMON=(
    --backend openai-chat
    --model "$MODEL"
    --base-url "http://127.0.0.1:${PORT}"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --multi-turn
    --multi-turn-num-turns "$MULTI_TURN_NUM_TURNS"
    --multi-turn-concurrency "$MULTI_TURN_CONCURRENCY"
    --multi-turn-delay-ms "$MULTI_TURN_DELAY_MS"
    --multi-turn-prefix-global-ratio "$GLOBAL_PREFIX_RATIO"
    --multi-turn-prefix-conversation-ratio "$CONV_PREFIX_RATIO"
    --percentile-metrics "ttft,tpot,itl,e2el"
    --save-result
    --result-dir "$RESULT_DIR"
)

wait_for_server() {
    echo "  Waiting for server on port $PORT..."
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            echo "  Server ready."
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: server did not start within 180s" >&2
    return 1
}

kill_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
    fi
}
trap kill_server EXIT

mooncake_owner_reminder() {
    echo "  Mooncake backend expects an external owner to be running before vLLM starts."
    echo "  setup_vllm_env.sh only configures requester-side environment."
}

run_one() {
    local label="$1"
    local result_file="$2"
    shift 2
    local server_extra=("$@")

    echo ""
    echo ">>> Starting server: $label"
    printf 'Command: vllm serve'
    printf ' %q' "${SERVER_COMMON[@]}" "${server_extra[@]}"
    printf '\n'
    vllm serve \
        "${SERVER_COMMON[@]}" "${server_extra[@]}" &
    SERVER_PID=$!

    wait_for_server

    echo ">>> Running benchmark: $label"
    vllm-bench \
        "${BENCH_COMMON[@]}" \
        --result-filename "$result_file"

    echo ">>> Stopping server"
    sleep 2
    kill_server
}

echo "============================================"
echo "  Multi-Turn Offloading Benchmark"
echo "============================================"
echo "  Model:         $MODEL"
echo "  Input len:     $INPUT_LEN"
echo "  Output len:    $OUTPUT_LEN"
echo "  Num prompts:   $NUM_PROMPTS"
echo "  Turns:         $MULTI_TURN_NUM_TURNS"
echo "  Concurrency:   $MULTI_TURN_CONCURRENCY"
echo "  Turn delay:    ${MULTI_TURN_DELAY_MS} ms"
echo "  Global prefix: $GLOBAL_PREFIX_RATIO"
echo "  Conv prefix:   $CONV_PREFIX_RATIO"
echo "  Backends:      $BACKENDS"
echo "  CPU offload:   ${CPU_OFFLOAD_GIB} GiB"
if [[ -n "$DISK_OFFLOAD_GIB" ]]; then
echo "  Disk offload:  ${DISK_OFFLOAD_GIB} GiB"
fi
echo "============================================"

# ── Run each requested backend ──────────────────────────────────
IFS=',' read -ra BACKEND_LIST <<< "$BACKENDS"

for backend in "${BACKEND_LIST[@]}"; do
    case "$backend" in
        baseline)
            run_one "Baseline (no offloading)" "mt_baseline.json"
            ;;
        native)
            export VLLM_USE_SIMPLE_KV_OFFLOAD=0
            run_one "With CPU offloading (native)" "mt_native.json" \
                --kv-offloading-size "$CPU_OFFLOAD_GIB" \
                --kv-offloading-backend "native"
            ;;
        simple)
            export VLLM_USE_SIMPLE_KV_OFFLOAD=1
            run_one "With CPU offloading (simple)" "mt_simple.json" \
                --kv-offloading-size "$CPU_OFFLOAD_GIB" \
                --kv-offloading-backend "native"
            ;;
        mooncake)
            export VLLM_USE_SIMPLE_KV_OFFLOAD=0
            SETUP_ARGS=(--cpu-mem-size "$CPU_OFFLOAD_GIB")
            if [[ -n "$DISK_OFFLOAD_GIB" ]]; then
                SETUP_ARGS+=(--disk-size "$DISK_OFFLOAD_GIB")
            fi
            source "${SCRIPT_DIR}/setup_vllm_env.sh" "${SETUP_ARGS[@]}"
            mooncake_owner_reminder
            run_one "With CPU offloading (mooncake)" "mt_mooncake.json" \
                --kv-transfer-config "{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"load_async\":true}}"
            ;;
        mooncake-mem)
            export VLLM_USE_SIMPLE_KV_OFFLOAD=0
            SETUP_ARGS=(--cpu-mem-size "$CPU_OFFLOAD_GIB")
            source "${SCRIPT_DIR}/setup_vllm_env.sh" "${SETUP_ARGS[@]}"
            mooncake_owner_reminder
            run_one "With CPU offloading (mooncake-mem)" "mt_mooncake_mem.json" \
                --kv-transfer-config "{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"load_async\":true}}"
            ;;
        *)
            echo ""
            echo ">>> Skipping unknown backend: $backend"
            ;;
    esac
done

# ── Compare all available results ───────────────────────────────
python "${SCRIPT_DIR}/compare_results.py" "$RESULT_DIR" --prefix mt_
