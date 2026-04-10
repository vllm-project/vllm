#!/usr/bin/env bash
# Benchmark CPU KV cache offloading overhead.
# Launches a vllm server, runs `vllm-bench` against it, then
# repeats with offloading enabled. Uses random (unique) prompts so
# no offloaded prefix is ever hit — isolating the pure store overhead.
#
# Usage:
#   bash benchmark_cpu_offloading.sh [MODEL] [INPUT_LEN] [OUTPUT_LEN] [NUM_PROMPTS]
#
# Supported backends (comma-separated in BACKENDS):
#   baseline        - No offloading
#   native          - Built-in vLLM KV offloading (--kv-offloading-backend native)
#   simple          - Simple native offload (VLLM_USE_SIMPLE_KV_OFFLOAD=1 + native backend)
#   mooncake        - MooncakeStoreConnector via --kv-transfer-config
#
# Environment variables:
#   CPU_OFFLOAD_GIB       - CPU offload buffer in GiB   (default: 80)
#   DISK_OFFLOAD_GIB      - Disk offload quota in GiB   (default: unset = disabled)
#   REQUEST_RATE          - Requests/s to the server     (default: 1)
#   PORT                  - Server port                  (default: 8192)
#   RESULT_DIR            - Output directory             (default: ./bench_results)
#   BACKENDS              - Comma-separated backends     (default: baseline,native,simple,mooncake)
#   MOONCAKE_CONFIG_PATH  - Path to mooncake config JSON (required for mooncake, auto-skipped if unset)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
# MODEL="${1:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
INPUT_LEN="${2:-8192}"
OUTPUT_LEN="${3:-1024}"
NUM_PROMPTS="${4:-200}"
CPU_OFFLOAD_GIB="${CPU_OFFLOAD_GIB:-80}"
DISK_OFFLOAD_GIB="${DISK_OFFLOAD_GIB:-400}"
REQUEST_RATE="${REQUEST_RATE:-1}"
PORT="${PORT:-8192}"
RESULT_DIR="${RESULT_DIR:-./bench_results}"
BACKENDS="${BACKENDS:-baseline,mooncake}"

mkdir -p "$RESULT_DIR"

if [[ -v MC_TCP_ENABLE_CONNECTION_POOL ]]; then
    ORIGINAL_MC_TCP_ENABLE_CONNECTION_POOL="$MC_TCP_ENABLE_CONNECTION_POOL"
    ORIGINAL_MC_TCP_ENABLE_CONNECTION_POOL_SET=1
else
    ORIGINAL_MC_TCP_ENABLE_CONNECTION_POOL=""
    ORIGINAL_MC_TCP_ENABLE_CONNECTION_POOL_SET=0
fi

if [[ -v MOONCAKE_CONFIG_PATH ]]; then
    ORIGINAL_MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG_PATH"
    ORIGINAL_MOONCAKE_CONFIG_PATH_SET=1
else
    ORIGINAL_MOONCAKE_CONFIG_PATH=""
    ORIGINAL_MOONCAKE_CONFIG_PATH_SET=0
fi

if [[ -v MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES ]]; then
    ORIGINAL_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES="$MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"
    ORIGINAL_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES_SET=1
else
    ORIGINAL_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=""
    ORIGINAL_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES_SET=0
fi

if [[ -v VLLM_USE_SIMPLE_KV_OFFLOAD ]]; then
    ORIGINAL_VLLM_USE_SIMPLE_KV_OFFLOAD="$VLLM_USE_SIMPLE_KV_OFFLOAD"
    ORIGINAL_VLLM_USE_SIMPLE_KV_OFFLOAD_SET=1
else
    ORIGINAL_VLLM_USE_SIMPLE_KV_OFFLOAD=""
    ORIGINAL_VLLM_USE_SIMPLE_KV_OFFLOAD_SET=0
fi

SERVER_COMMON=(
    --model "$MODEL"
    --disable-hybrid-kv-cache-manager
    --port "$PORT"
    --no-enable-log-requests
    --attention-backend FLASHINFER
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
    --model "$MODEL"
    --base-url "http://127.0.0.1:${PORT}"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$REQUEST_RATE"
    --seed 42
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

restore_env_var() {
    local var_name="$1"
    local was_set="$2"
    local value="$3"

    if [[ "$was_set" == "1" ]]; then
        export "$var_name=$value"
    else
        unset "$var_name"
    fi
}

reset_backend_env() {
    restore_env_var "MC_TCP_ENABLE_CONNECTION_POOL" \
        "$ORIGINAL_MC_TCP_ENABLE_CONNECTION_POOL_SET" \
        "$ORIGINAL_MC_TCP_ENABLE_CONNECTION_POOL"
    restore_env_var "MOONCAKE_CONFIG_PATH" \
        "$ORIGINAL_MOONCAKE_CONFIG_PATH_SET" \
        "$ORIGINAL_MOONCAKE_CONFIG_PATH"
    restore_env_var "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES" \
        "$ORIGINAL_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES_SET" \
        "$ORIGINAL_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"
    restore_env_var "VLLM_USE_SIMPLE_KV_OFFLOAD" \
        "$ORIGINAL_VLLM_USE_SIMPLE_KV_OFFLOAD_SET" \
        "$ORIGINAL_VLLM_USE_SIMPLE_KV_OFFLOAD"
}

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
echo "  CPU Offloading Overhead Benchmark"
echo "============================================"
echo "  Model:         $MODEL"
echo "  Input len:     $INPUT_LEN"
echo "  Output len:    $OUTPUT_LEN"
echo "  Num prompts:   $NUM_PROMPTS"
echo "  Request rate:  $REQUEST_RATE req/s"
echo "  Backends:      $BACKENDS"
echo "  CPU offload:   ${CPU_OFFLOAD_GIB} GiB"
if [[ -n "$DISK_OFFLOAD_GIB" ]]; then
echo "  Disk offload:  ${DISK_OFFLOAD_GIB} GiB"
fi
echo "============================================"

# ── Run each requested backend ──────────────────────────────────
IFS=',' read -ra BACKEND_LIST <<< "$BACKENDS"

for backend in "${BACKEND_LIST[@]}"; do
    reset_backend_env
    case "$backend" in
        baseline)
            run_one "Baseline (no offloading)" "baseline.json"
            ;;
        native)
            export VLLM_USE_SIMPLE_KV_OFFLOAD=0
            run_one "With CPU offloading (native)" "native.json" \
                --kv-offloading-size "$CPU_OFFLOAD_GIB" \
                --kv-offloading-backend "native"
            ;;
        simple)
            export VLLM_USE_SIMPLE_KV_OFFLOAD=1
            run_one "With CPU offloading (simple)" "simple.json" \
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
            run_one "With CPU offloading (mooncake)" "mooncake.json" \
                --kv-transfer-config "{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"load_async\":true}}"
            ;;
        *)
            echo ""
            echo ">>> Skipping unknown backend: $backend"
            ;;
    esac
done

# ── Compare all available results ───────────────────────────────
python3 "${SCRIPT_DIR}/compare_results.py" "$RESULT_DIR"
