#!/usr/bin/env bash
# Benchmark CPU KV cache offloading overhead.
# Launches a vllm server, runs `vllm bench serve` against it, then
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
#   KV_OFFLOAD_GIB        - CPU offload buffer in GiB   (default: 80)
#   REQUEST_RATE          - Requests/s to the server     (default: 1)
#   PORT                  - Server port                  (default: 8192)
#   RESULT_DIR            - Output directory             (default: ./bench_results)
#   BACKENDS              - Comma-separated backends     (default: baseline,native,simple,mooncake)
#   MOONCAKE_CONFIG_PATH  - Path to mooncake config JSON (required for mooncake, auto-skipped if unset)

set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B}"
# MODEL="${1:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
INPUT_LEN="${2:-8192}"
OUTPUT_LEN="${3:-1024}"
NUM_PROMPTS="${4:-200}"
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-80}"
REQUEST_RATE="${REQUEST_RATE:-1}"
PORT="${PORT:-8192}"
RESULT_DIR="${RESULT_DIR:-./bench_results}"
BACKENDS="${BACKENDS:-baseline,mooncake}"

mkdir -p "$RESULT_DIR"

SERVER_COMMON=(
    --model "$MODEL"
    --disable-hybrid-kv-cache-manager
    --port "$PORT"
    --no-enable-log-requests
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

run_one() {
    local label="$1"
    local result_file="$2"
    shift 2
    local server_extra=("$@")

    echo ""
    echo ">>> Starting server: $label"
    echo "Command: vllm serve ${SERVER_COMMON[@]} ${server_extra[@]}"
    vllm serve \
        "${SERVER_COMMON[@]}" "${server_extra[@]}" &
    SERVER_PID=$!

    wait_for_server

    echo ">>> Running benchmark: $label"
    vllm bench serve \
        "${BENCH_COMMON[@]}" \
        --result-filename "$result_file"

    echo ">>> Stopping server"
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
echo "  Offload size:  ${KV_OFFLOAD_GIB} GiB"
echo "============================================"

# ── Run each requested backend ──────────────────────────────────
IFS=',' read -ra BACKEND_LIST <<< "$BACKENDS"

for backend in "${BACKEND_LIST[@]}"; do
    case "$backend" in
        baseline)
            run_one "Baseline (no offloading)" "baseline.json"
            ;;
        native)
            export VLLM_USE_SIMPLE_KV_OFFLOAD=0
            run_one "With CPU offloading (native)" "native.json" \
                --kv-offloading-size "$KV_OFFLOAD_GIB" \
                --kv-offloading-backend "native"
            ;;
        simple)
            export VLLM_USE_SIMPLE_KV_OFFLOAD=1
            run_one "With CPU offloading (simple)" "simple.json" \
                --kv-offloading-size "$KV_OFFLOAD_GIB" \
                --kv-offloading-backend "native"
            ;;
        mooncake)
            if [[ -z "${MOONCAKE_CONFIG_PATH:-}" || ! -f "${MOONCAKE_CONFIG_PATH:-}" ]]; then
                echo ""
                echo ">>> Skipping mooncake: MOONCAKE_CONFIG_PATH not set or file missing"
                continue
            fi
            export VLLM_USE_SIMPLE_KV_OFFLOAD=0
            export MC_TCP_ENABLE_CONNECTION_POOL=1
            # Update mooncake config with the requested offload size
            python3 -c "
import json, sys

cfg_path = sys.argv[1]
size = sys.argv[2] + 'GB'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['global_segment_size'] = size
cfg['local_buffer_size'] = size

with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
    f.write('\n')
" "$MOONCAKE_CONFIG_PATH" "$KV_OFFLOAD_GIB"
            echo "  Updated $MOONCAKE_CONFIG_PATH: segment/buffer size = ${KV_OFFLOAD_GIB}GB"
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
python3 - "$RESULT_DIR" <<'PYEOF'
import json, sys, os

d = sys.argv[1]

# Backend label -> result filename
ALL_BACKENDS = [
    ("baseline", "baseline.json"),
    ("native",   "native.json"),
    ("simple",   "simple.json"),
    ("mooncake", "mooncake.json"),
]

results = {}
for label, fname in ALL_BACKENDS:
    path = os.path.join(d, fname)
    if os.path.isfile(path):
        with open(path) as f:
            results[label] = json.load(f)

if not results:
    print("\nNo result files found — nothing to compare.\n")
    sys.exit(0)

metrics = [
    ("Output throughput (tok/s)", "output_throughput",  ".1f"),
    ("Request throughput (req/s)", "request_throughput", ".3f"),
    ("Mean TTFT (ms)",            "mean_ttft_ms",       ".2f"),
    ("Mean TPOT (ms)",            "mean_tpot_ms",       ".2f"),
]

labels = list(results.keys())
col_w = 14

def pct(old, new):
    return (new - old) / old * 100 if old else 0.0

print()
print("=" * (32 + col_w * len(labels)))
print("  COMPARISON  (delta vs baseline: + = regression, - = improvement)")
print("=" * (32 + col_w * len(labels)))

# Header
header = f"  {'Metric':<30}"
for lb in labels:
    header += f"  {lb:>{col_w - 2}}"
print(header)
sep = f"  {'-' * 30}"
for _ in labels:
    sep += f"  {'-' * (col_w - 2)}"
print(sep)

baseline = results.get("baseline")

for metric_label, key, fmt in metrics:
    row = f"  {metric_label:<30}"
    for lb in labels:
        val = results[lb][key]
        cell = f"{val:{fmt}}"
        if baseline and lb != "baseline":
            delta = pct(baseline[key], val)
            cell += f" ({delta:+.1f}%)"
        row += f"  {cell:>{col_w - 2}}"
    print(row)

print("=" * (32 + col_w * len(labels)))
print()
PYEOF
