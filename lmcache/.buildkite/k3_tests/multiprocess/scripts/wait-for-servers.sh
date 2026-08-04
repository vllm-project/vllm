#!/usr/bin/env bash
# Wait for vLLM servers to be ready (native processes, no Docker).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
BUILD_ID="${BUILD_ID:-local_$$}"

# Wait for a vLLM server with health check
wait_for_vllm_server() {
    local port="$1"
    local description="$2"
    local logfile="$3"

    echo "=== Waiting for $description to be ready ==="
    echo "Port: $port, Max wait: ${MAX_WAIT_SECONDS}s"

    local start_time end_time
    start_time=$(date +%s)
    end_time=$((start_time + MAX_WAIT_SECONDS))

    while true; do
        local current_time elapsed
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        if [ "$current_time" -ge "$end_time" ]; then
            echo "Timeout: $description did not become ready within ${MAX_WAIT_SECONDS}s"
            echo ""
            echo "=== $description log (last 100 lines) ==="
            tail -100 "$logfile" 2>/dev/null || true
            return 1
        fi

        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "$description is ready! (took ${elapsed}s)"
            return 0
        fi

        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "$description is ready! (took ${elapsed}s)"
            return 0
        fi

        echo "Waiting for $description... (${elapsed}s elapsed)"
        sleep 5
    done
}

# Wait for both servers (they start simultaneously)
if ! wait_for_vllm_server "$VLLM_PORT" "vLLM with LMCache" \
        "/tmp/build_${BUILD_ID}_vllm.log"; then
    exit 1
fi

# The baseline server only exists for 2-GPU tests; 1-GPU tests set
# LAUNCH_BASELINE=false in launch-processes.sh and never start it.
if [[ "${LAUNCH_BASELINE:-true}" == "true" ]]; then
    if ! wait_for_vllm_server "$VLLM_BASELINE_PORT" "vLLM baseline (without LMCache)" \
            "/tmp/build_${BUILD_ID}_vllm_baseline.log"; then
        exit 1
    fi
fi

echo ""
echo "=== All vLLM servers are ready ==="
