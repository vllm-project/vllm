#!/usr/bin/env bash
# Main orchestrator for multiprocessing tests (native, no Docker).
# Launches LMCache MP server + vLLM with LMCache + vLLM baseline,
# then runs workloads (lm_eval, vllm bench, long_doc_qa).
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

# ── Configuration ────────────────────────────────────────────
export LMCACHE_PORT="${LMCACHE_PORT:-6555}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"
export BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
export MODEL="${MODEL:-Qwen/Qwen3-14B}"
export CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
export MAX_WORKERS="${MAX_WORKERS:-4}"
export LMCACHE_DIR="$REPO_ROOT"
export RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

mkdir -p "$RESULTS_DIR"

TEST_RESULT=0

# Cleanup: always kill background processes on exit
trap '"${SCRIPT_DIR}/cleanup.sh"' EXIT

echo "============================================"
echo "=== LMCache Multiprocessing Test ==="
echo "============================================"
echo "Build ID: $BUILD_ID"
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT"
echo "vLLM port: $VLLM_PORT"
echo "vLLM baseline port: $VLLM_BASELINE_PORT"
echo "Results dir: $RESULTS_DIR"
echo ""

# Step 1: Launch native processes (replaces Docker containers)
echo "============================================"
echo "=== Step 1: Launching native processes ==="
echo "============================================"
if ! "${SCRIPT_DIR}/launch-processes.sh"; then
    echo "Failed to launch processes"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 2: Wait for vLLM to be ready
echo "============================================"
echo "=== Step 2: Waiting for vLLM to be ready ==="
echo "============================================"
if ! "${SCRIPT_DIR}/wait-for-servers.sh"; then
    echo "vLLM failed to become ready"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 3: Run lm_eval workload test
echo "============================================"
echo "=== Step 3: Running lm_eval workload ==="
echo "============================================"
if ! "${SCRIPT_DIR}/run-lm-eval.sh"; then
    echo "lm_eval workload test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 4: Run vllm bench serve test
echo "============================================"
echo "=== Step 4: Running vllm bench serve ==="
echo "============================================"
if ! "${SCRIPT_DIR}/run-vllm-bench.sh"; then
    echo "vllm bench serve test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 5: Run long doc QA test
echo "============================================"
echo "=== Step 5: Running long doc QA ==="
echo "============================================"
if ! "${SCRIPT_DIR}/run-long-doc-qa.sh"; then
    echo "long doc QA test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 6: L2 long doc QA test (restarts LMCache with L2 config)
echo "============================================"
echo "=== Step 6: Running long doc QA (L2) ==="
echo "============================================"
if ! "${SCRIPT_DIR}/run-long-doc-qa-l2.sh"; then
    echo "long doc QA L2 test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 7: Fault tolerance test (kills LMCache server -- must be last)
echo "============================================"
echo "=== Step 7: Running fault tolerance test ==="
echo "============================================"
if ! "${SCRIPT_DIR}/run-fault-tolerance.sh"; then
    echo "fault tolerance test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

echo "============================================"
echo "=== All tests passed! ==="
echo "============================================"
