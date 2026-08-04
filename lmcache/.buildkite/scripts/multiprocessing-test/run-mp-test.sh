#!/bin/bash
# Main orchestrator script for multiprocessing tests
# This script ensures cleanup always runs, even on failure

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export container names and configuration for all scripts
export LMCACHE_CONTAINER_NAME="lmcache-mp-test-$$"
export VLLM_CONTAINER_NAME="vllm-mp-test-$$"
export VLLM_BASELINE_CONTAINER_NAME="vllm-baseline-test-$$"
export LMCACHE_PORT="${LMCACHE_PORT:-6555}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"
export BUILD_ID="${BUILD_ID:-local_$$}"

# Track the overall test result
TEST_RESULT=0

# Cleanup function - always runs on exit
cleanup() {
    echo ""
    echo "============================================"
    echo "=== Running cleanup (exit code: $TEST_RESULT) ==="
    echo "============================================"
    "$SCRIPT_DIR/cleanup.sh"
}

# Register cleanup to run on exit (covers normal exit, errors, and signals)
trap cleanup EXIT

echo "============================================"
echo "=== LMCache Multiprocessing Test ==="
echo "============================================"
echo "Build ID: $BUILD_ID"
echo "LMCache container: $LMCACHE_CONTAINER_NAME"
echo "vLLM container: $VLLM_CONTAINER_NAME"
echo "vLLM baseline container: $VLLM_BASELINE_CONTAINER_NAME"
echo "LMCache port: $LMCACHE_PORT"
echo "vLLM port: $VLLM_PORT"
echo "vLLM baseline port: $VLLM_BASELINE_PORT"
echo ""

# Step 1: Build docker images
echo "============================================"
echo "=== Step 1: Building Docker images ==="
echo "============================================"
if ! "$SCRIPT_DIR/build-mp-docker-image.sh"; then
    echo "❌ Failed to build docker images"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 2: Launch containers
echo "============================================"
echo "=== Step 2: Launching containers ==="
echo "============================================"
if ! "$SCRIPT_DIR/launch-containers.sh"; then
    echo "❌ Failed to launch containers"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 3: Wait for vLLM to be ready
echo "============================================"
echo "=== Step 3: Waiting for vLLM to be ready ==="
echo "============================================"
if ! "$SCRIPT_DIR/wait-for-vllm.sh"; then
    echo "❌ vLLM failed to become ready"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 4: Run lm_eval workload test
echo "============================================"
echo "=== Step 4: Running lm_eval workload ==="
echo "============================================"
if ! "$SCRIPT_DIR/run-lm-eval.sh"; then
    echo "❌ lm_eval workload test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 5: Run vllm bench serve test
echo "============================================"
echo "=== Step 5: Running vllm bench serve ==="
echo "============================================"
if ! "$SCRIPT_DIR/run-vllm-bench.sh"; then
    echo "❌ vllm bench serve test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 6: Run long doc QA test
echo "============================================"
echo "=== Step 6: Running long doc QA ==="
echo "============================================"
if ! "$SCRIPT_DIR/run-long-doc-qa.sh"; then
    echo "❌ long doc QA test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

# Step 7: Run fault tolerance test (destructive — kills LMCache container)
echo "============================================"
echo "=== Step 7: Running fault tolerance test ==="
echo "============================================"
if ! "$SCRIPT_DIR/run-fault-tolerance.sh"; then
    echo "❌ fault tolerance test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""

echo "============================================"
echo "=== ✅ All tests passed! ==="
echo "============================================"

# Step 8: Cleanup runs automatically via trap

