#!/usr/bin/env bash
# Benchmark CPU KV cache offloading overhead.
# Launches a vllm server, runs `vllm bench serve` against it, then
# repeats with offloading enabled. Uses random (unique) prompts so
# no offloaded prefix is ever hit — isolating the pure store overhead.
#
# Usage:
#   bash benchmarks/benchmark_cpu_offloading.sh [MODEL] [INPUT_LEN] [OUTPUT_LEN] [NUM_PROMPTS]
#
# Environment variables:
#   KV_OFFLOAD_GIB  - CPU offload buffer in GiB   (default: 40)
#   REQUEST_RATE    - Requests/s to the server     (default: 8)
#   PORT            - Server port                  (default: 8192)
#   RESULT_DIR      - Output directory             (default: ./bench_results)
#   OFFLOAD_BACKEND - Offloading backend            (default: native)
#   RUN_BASELINE    - Run baseline phase (1/0)      (default: 1)
#   RUN_OFFLOADING  - Run offloading phase (1/0)    (default: 1)

set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B}"
# MODEL="${1:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
INPUT_LEN="${2:-8192}"
OUTPUT_LEN="${3:-1024}"
NUM_PROMPTS="${4:-500}"
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-80}"
REQUEST_RATE="${REQUEST_RATE:-3}"
PORT="${PORT:-8192}"
RESULT_DIR="${RESULT_DIR:-./bench_results}"
OFFLOAD_BACKEND="${OFFLOAD_BACKEND:-simple}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_OFFLOADING="${RUN_OFFLOADING:-1}"

mkdir -p "$RESULT_DIR"

SERVER_COMMON=(
    --model "$MODEL"
    --no-disable-hybrid-kv-cache-manager
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
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            echo "  Server ready."
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: server did not start within 120s" >&2
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
    # VLLM_PROFILE_KV_CONNECTOR="${VLLM_PROFILE_KV_CONNECTOR:-0}" \
    # VLLM_PROFILE_KV_INTERVAL="${VLLM_PROFILE_KV_INTERVAL:-50}" \
    # nsys profile --delay 130 --duration 30 --sample none --trace cuda,nvtx,cudnn,cublas -o "$label-$MODEL" -- \
    numactl --cpunodebind=0 --membind=0 \
    python -m vllm.entrypoints.openai.api_server \
        "${SERVER_COMMON[@]}" "${server_extra[@]}" &
    SERVER_PID=$!

    wait_for_server

    echo ">>> Running benchmark: $label"
    python -m vllm.entrypoints.cli.main bench serve \
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
echo "  Offload size:  ${KV_OFFLOAD_GIB} GiB"
echo "============================================"

if [[ "$RUN_BASELINE" == "1" ]]; then
    run_one "Baseline (no offloading)" "baseline.json"
fi

        # --no-disable-hybrid-kv-cache-manager \
if [[ "$RUN_OFFLOADING" == "1" ]]; then
    run_one "With CPU offloading" "offloading.json" \
        --kv-offloading-size "$KV_OFFLOAD_GIB" \
        --kv-offloading-backend "$OFFLOAD_BACKEND"
fi

# ── Compare results ──────────────────────────────────────────────
python3 - "$RESULT_DIR" <<'PYEOF'
import json, sys

d = sys.argv[1]
with open(f"{d}/baseline.json") as f:
    base = json.load(f)
with open(f"{d}/offloading.json") as f:
    offl = json.load(f)

def pct(old, new):
    return (new - old) / old * 100 if old else 0.0

b_tps  = base["output_throughput"]
o_tps  = offl["output_throughput"]
b_rps  = base["request_throughput"]
o_rps  = offl["request_throughput"]
b_ttft = base["mean_ttft_ms"]
o_ttft = offl["mean_ttft_ms"]
b_tpot = base["mean_tpot_ms"]
o_tpot = offl["mean_tpot_ms"]

print()
print("=" * 72)
print("  COMPARISON  (+ = regression,  - = improvement)")
print("=" * 72)
fmt = "  {:<30} {:>12} {:>12} {:>12}"
print(fmt.format("Metric", "Baseline", "Offloading", "Delta"))
print(fmt.format("-" * 30, "-" * 12, "-" * 12, "-" * 12))
print(fmt.format("Output throughput (tok/s)",
                  f"{b_tps:.1f}", f"{o_tps:.1f}", f"{pct(b_tps, o_tps):+.2f}%"))
print(fmt.format("Request throughput (req/s)",
                  f"{b_rps:.3f}", f"{o_rps:.3f}", f"{pct(b_rps, o_rps):+.2f}%"))
print(fmt.format("Mean TTFT (ms)",
                  f"{b_ttft:.2f}", f"{o_ttft:.2f}", f"{pct(b_ttft, o_ttft):+.2f}%"))
print(fmt.format("Mean TPOT (ms)",
                  f"{b_tpot:.2f}", f"{o_tpot:.2f}", f"{pct(b_tpot, o_tpot):+.2f}%"))
print("=" * 72)
print()
PYEOF
