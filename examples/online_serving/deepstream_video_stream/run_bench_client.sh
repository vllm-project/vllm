#!/bin/bash
# ============================================================================
# Run bench_h264_client.py against a running vLLM server.
# Start the server first with start_server.sh.
#
# Usage:
#   ./run_bench_client.sh                           # defaults
#   ./run_bench_client.sh --num-prompts 50          # override prompts
#   ./run_bench_client.sh --request-rate inf        # all at once
#
# All arguments are passed through to bench_h264_client.py.
# If no --video is given, defaults to /data/video/drivesim.mp4.
#
# VIDEO can be:
#   - A single file:   VIDEO=/data/video/drivesim.mp4
#   - A directory:      VIDEO=/data/video/        (all mp4/mkv/avi/ts/mov)
#   - Multiple files:   VIDEO="/data/video/a.mp4 /data/video/b.mp4"
#
# With multiple videos, requests are distributed round-robin.
#
# Environment overrides:
#   PORT=8000  VIDEO=/data/video/  BACKEND_LABEL=deepstream
#   NUM_PROMPTS=100  REQUEST_RATE=16  MAX_TOKENS=16
# ============================================================================
set -euo pipefail

ulimit -n 65535

PORT="${PORT:-8000}"
VIDEO="${VIDEO:-/data/video/}"
BACKEND_LABEL="${BACKEND_LABEL:-unknown}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
REQUEST_RATE="${REQUEST_RATE:-16}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CHUNK_DURATION="${CHUNK_DURATION:-10}"
BENCH_CLIENT="$(dirname "$0")/bench_h264_client.py"
RESULTS_DIR="/work/deepstream_9.0_vllm/bench_results"
RESULT_FILE="${RESULTS_DIR}/bench_serve_${BACKEND_LABEL}.json"
LOG_FILE="${RESULTS_DIR}/bench_serve_${BACKEND_LABEL}.log"

mkdir -p "$RESULTS_DIR"

# Check server is running
echo "  Checking server at http://localhost:$PORT ..."
if ! curl -s --max-time 5 http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "  ERROR: Server not reachable at http://localhost:$PORT"
    echo "  Start it first:  ./start_server.sh deepstream"
    exit 1
fi
echo "  Server is healthy."
echo ""

echo "============================================================"
echo "  Benchmark Client"
echo "============================================================"
echo "  Video          : $VIDEO"
echo "  Server         : http://localhost:$PORT"
echo "  Backend label  : $BACKEND_LABEL"
echo "  Num prompts    : $NUM_PROMPTS"
echo "  Request rate   : $REQUEST_RATE"
echo "  Max tokens     : $MAX_TOKENS"
echo "  Chunk duration : ${CHUNK_DURATION}s"
echo "  Result file    : $RESULT_FILE"
echo "============================================================"
echo ""

# Build default args, then allow overrides via "$@"
# shellcheck disable=SC2086
DEFAULT_ARGS=(
    --video $VIDEO
    --base-url "http://localhost:$PORT"
    --model bench-model
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$REQUEST_RATE"
    --max-tokens "$MAX_TOKENS"
    --chunk-duration "$CHUNK_DURATION"
    --force
    --use-file-url
    --save-result
    --result-filename "$RESULT_FILE"
)

python3 "$BENCH_CLIENT" "${DEFAULT_ARGS[@]}" "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "  Results saved to: $RESULT_FILE"
echo "  Log saved to:     $LOG_FILE"
echo ""
