#!/usr/bin/env bash
# Cleanup background processes launched for multiprocessing tests.
# This script should always be called, even on failure.

BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"

echo "=== Cleaning up background processes ==="

if [[ -f "$PID_FILE" ]]; then
    while IFS= read -r pid; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            # Capture a snippet of the process log before killing
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            echo "PID $pid stopped"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
else
    echo "No PID file found at $PID_FILE"
fi

# Also kill any stray vllm/lmcache processes from this build
# (safety net in case PIDs weren't recorded)
for port in "${VLLM_PORT:-8000}" "${VLLM_BASELINE_PORT:-9000}" "${LMCACHE_PORT:-6555}"; do
    fuser -k "${port}/tcp" 2>/dev/null || true
done

# Remove the GDS slab scratch dir (only set for gds_* tests). It lives on the
# /scratch hostPath (host-local NVMe), so it persists past the pod and the
# preallocated slab is large -- drop it now that the server is stopped.
if [[ -n "${GDS_L1_PATH:-}" ]]; then
    echo "Removing GDS slab dir: $GDS_L1_PATH"
    rm -rf "${GDS_L1_PATH}" 2>/dev/null || true
fi

echo "=== Cleanup complete ==="

# Copy server logs to the workspace so Buildkite can collect them as artifacts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cp /tmp/build_${BUILD_ID}_*.log "${REPO_ROOT}/" 2>/dev/null || true

# Wait for GPU memory to be fully released
echo "Waiting 5 seconds for GPU memory to be released..."
sleep 5
