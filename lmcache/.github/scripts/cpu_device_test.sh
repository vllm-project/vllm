#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# CPU device test: runs both server bench and vLLM e2e tests
# in the same environment to avoid repeated vLLM installation.
#
# Usage: cpu_device_test.sh [mode]
#   mode: server_bench, vllm_e2e, or all (default)
#
# Environment variables:
#   LMCACHE_BENCH_TRANSFER_MODE  engine_driven|lmcache_driven
#                                (default: engine_driven)
#   LMCACHE_E2E_TRANSPORT_MODE   engine_driven|lmcache_driven|shm|pickle
#                                (default: engine_driven)
#   LMCACHE_E2E_DATA_MODE        shm|pickle (default: shm)
#   LMCACHE_HTTP_PORT_BENCH      HTTP port for bench (default: 18080)
#   LMCACHE_ZMQ_PORT_BENCH       ZMQ port for bench (default: 15555)
#   LMCACHE_HTTP_PORT_E2E        HTTP port for e2e (default: 18081)
#   LMCACHE_ZMQ_PORT_E2E         ZMQ port for e2e (default: 15557)
#   VLLM_PORT_E2E                HTTP port for vLLM (default: 18000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"
TEST_MODE="${1:-all}"

echo "==> CPU device test (OS: ${OS}, Mode: ${TEST_MODE})"
echo "    Python: $(python3 --version 2>&1 || true)"

# Configuration
BENCH_TRANSFER_MODE="${LMCACHE_BENCH_TRANSFER_MODE:-engine_driven}"
E2E_TRANSPORT_MODE="${LMCACHE_E2E_TRANSPORT_MODE:-engine_driven}"
E2E_DATA_MODE="${LMCACHE_E2E_DATA_MODE:-shm}"
HTTP_PORT_BENCH="${LMCACHE_HTTP_PORT_BENCH:-18080}"
ZMQ_PORT_BENCH="${LMCACHE_ZMQ_PORT_BENCH:-15555}"
HTTP_PORT_E2E="${LMCACHE_HTTP_PORT_E2E:-18081}"
ZMQ_PORT_E2E="${LMCACHE_ZMQ_PORT_E2E:-15557}"
VLLM_PORT_E2E="${VLLM_PORT_E2E:-18000}"

# Validate modes
case "${BENCH_TRANSFER_MODE}" in
  lmcache_driven|engine_driven) ;;
  *)
    echo "!! Unknown LMCACHE_BENCH_TRANSFER_MODE='${BENCH_TRANSFER_MODE}'"
    exit 1
    ;;
esac

# Map user-facing LMCACHE_E2E_TRANSPORT_MODE to internal representation.
# shm/pickle are aliases for lmcache_driven mode with sub-mode selection.
case "${E2E_TRANSPORT_MODE}" in
  lmcache_driven|engine_driven)
    MAPPED_TRANSPORT_MODE="${E2E_TRANSPORT_MODE}"
    MAPPED_DATA_MODE="${E2E_DATA_MODE}"
    ;;
  shm)
    MAPPED_TRANSPORT_MODE="engine_driven"
    MAPPED_DATA_MODE="shm"
    ;;
  pickle)
    MAPPED_TRANSPORT_MODE="engine_driven"
    MAPPED_DATA_MODE="pickle"
    ;;
  *)
    echo "!! Unknown LMCACHE_E2E_TRANSPORT_MODE='${E2E_TRANSPORT_MODE}'"
    echo "   Valid values: engine_driven, lmcache_driven, shm, pickle"
    exit 1
    ;;
esac

echo "    Bench transfer mode: ${BENCH_TRANSFER_MODE}"
echo "    E2E transport mode: ${E2E_TRANSPORT_MODE}"
echo "    Ports: bench=${HTTP_PORT_BENCH}/${ZMQ_PORT_BENCH}, e2e=${HTTP_PORT_E2E}/${ZMQ_PORT_E2E}/${VLLM_PORT_E2E}"

# Reap any LMCache/vLLM children started by this run on exit so the
# next workflow step does not collide on default ZMQ/HTTP ports.
cleanup_processes_safe() {
    local rc=$?
    set +e
    # Kill child processes started by this shell first (e.g. lmcache server
    # backgrounded by the shared validation script).
    pkill -P $$ 2>/dev/null || true
    sleep 1
    pkill -9 -P $$ 2>/dev/null || true
    return $rc
}
trap cleanup_processes_safe EXIT

# Function to run server bench test
run_server_bench() {
    echo ""
    echo "==> Running CPU server bench test"
    
    # Set environment for bench test
    export LMCACHE_BENCH_TRANSFER_MODE="${BENCH_TRANSFER_MODE}"
    export LMCACHE_HTTP_PORT="${HTTP_PORT_BENCH}"
    export LMCACHE_ZMQ_PORT="${ZMQ_PORT_BENCH}"
    export LMCACHE_LOG_FILE="/tmp/cpu_device_bench_${BENCH_TRANSFER_MODE}_lmcache.log"
    export BENCH_OUTPUT_LOG="/tmp/cpu_device_bench_${BENCH_TRANSFER_MODE}_output.log"
    export LMCACHE_HEALTHCHECK_TIMEOUT="30"
    export BENCH_NUM_REQUESTS="3"
    export BENCH_NUM_TOKENS="512"
    
    # Run bench test
    bash "${SCRIPT_DIR}/cpu_server_bench_test.sh"
    
    echo "==> CPU server bench test completed successfully"
}

# Function to run vLLM e2e test
run_vllm_e2e() {
    echo ""
    echo "==> Running CPU vLLM e2e test"
    
    # Set environment for e2e test
    export LMCACHE_TRANSPORT_MODE="${MAPPED_TRANSPORT_MODE}"
    export LMCACHE_DATA_MODE="${MAPPED_DATA_MODE}"
    export LMCACHE_HTTP_PORT="${HTTP_PORT_E2E}"
    export LMCACHE_ZMQ_PORT="${ZMQ_PORT_E2E}"
    export VLLM_PORT="${VLLM_PORT_E2E}"
    export LMCACHE_LOG_FILE="/tmp/cpu_device_e2e_${E2E_TRANSPORT_MODE}_lmcache.log"
    export VLLM_LOG_FILE="/tmp/cpu_device_e2e_${E2E_TRANSPORT_MODE}_vllm.log"
    export LMCACHE_HEALTHCHECK_TIMEOUT="30"
    export VLLM_READY_TIMEOUT="300"
    
    # Run e2e test
    bash "${SCRIPT_DIR}/cpu_vllm_e2e_test.sh"
    
    echo "==> CPU vLLM e2e test completed successfully"
}

# Determine which tests to run
case "${TEST_MODE}" in
    "server_bench")
        run_server_bench
        ;;
    "vllm_e2e")
        run_vllm_e2e
        ;;
    "all")
        run_server_bench
        run_vllm_e2e
        ;;
    *)
        echo "!! Unknown test mode: ${TEST_MODE}"
        echo "    Supported modes: server_bench, vllm_e2e, all"
        exit 1
        ;;
esac

echo ""
echo "==> CPU device test passed for modes:"
echo "    Test mode: ${TEST_MODE}"
echo "    Server bench: ${BENCH_TRANSFER_MODE}"
echo "    vLLM e2e: ${E2E_TRANSPORT_MODE}"
