#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Setup script for Prime-RL integration tests
# This script prepares the environment for running Prime-RL tests with nightly vLLM
# Enhanced with comprehensive diagnostics to debug test failures

set -euo pipefail

# Error trap to capture state on any failure
cleanup_and_diagnose() {
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "=== FAILURE DETECTED (exit code: $exit_code) ==="
    echo "=========================================="
    echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "Failed at line ${BASH_LINENO[0]}"
    
    # Capture all running processes
    echo ""
    echo "=== All Processes ==="
    ps aux | head -50 || true
    
    # Capture GPU state
    echo ""
    echo "=== GPU State ==="
    nvidia-smi || echo "nvidia-smi not available"
    
    # Capture disk usage
    echo ""
    echo "=== Disk Usage ==="
    df -h || true
    
    echo ""
    echo "=== Memory Usage ==="
    free -h || true
    
    # Check for any lingering vLLM processes
    echo ""
    echo "=== Cleaning up vLLM processes ==="
    pkill -9 vllm 2>/dev/null && echo "Killed lingering vLLM processes" || echo "No vLLM processes to kill"
  fi
}

trap cleanup_and_diagnose EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRIME_RL_REPO="https://github.com/PrimeIntellect-ai/prime-rl.git"
PRIME_RL_DIR="${REPO_ROOT}/prime-rl"

echo "=========================================="
echo "=== Prime-RL Integration Test Setup ==="
echo "=========================================="
echo "Start Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# ==========================================
# Environment Diagnostics
# ==========================================
echo ""
echo "=== Environment Diagnostics ==="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo "Script Directory: ${SCRIPT_DIR}"
echo "Repo Root: ${REPO_ROOT}"

echo ""
echo "=== Python Environment ==="
which python3 || echo "python3 not found in PATH"
python3 --version || echo "Cannot get python3 version"
which uv || echo "uv not found in PATH"
uv --version 2>/dev/null || echo "Cannot get uv version (will install)"

echo ""
echo "=== System Resources ==="
echo "Memory:"
free -h || true
echo ""
echo "Disk space:"
df -h . || true

echo ""
echo "=== GPU Information ==="
nvidia-smi || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Number of GPUs detected: $(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo '0')"

# ==========================================
# Setup Prime-RL Environment
# ==========================================
echo ""
echo "=========================================="
echo "=== Setting up Prime-RL Environment ==="
echo "=========================================="

# Clean up any existing Prime-RL directory
if [ -d "${PRIME_RL_DIR}" ]; then
    echo "Removing existing Prime-RL directory..."
    rm -rf "${PRIME_RL_DIR}"
fi

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    echo "UV installed. Version: $(uv --version)"
fi

# Clone Prime-RL repository at specific branch for reproducible tests
PRIME_RL_BRANCH="integ-vllm-main"
echo ""
echo "Cloning Prime-RL repository at branch: ${PRIME_RL_BRANCH}..."
git clone --branch "${PRIME_RL_BRANCH}" --single-branch "${PRIME_RL_REPO}" "${PRIME_RL_DIR}"
cd "${PRIME_RL_DIR}"

echo ""
echo "=== Prime-RL Repository Info ==="
echo "Current directory: $(pwd)"
echo "Git commit:"
git log -1 --oneline
echo "Git status:"
git status --short

echo ""
echo "Installing Prime-RL as a package..."
echo ""
echo "NOTE: Following CI pattern - using 'pip install' instead of 'uv sync'"
echo "      This preserves the Docker image's PyTorch version to maintain vLLM binary compatibility"

# Remove vllm pin from pyproject.toml to use pre-installed vLLM
echo ""
echo "Removing vllm pin from pyproject.toml..."
sed -i '/vllm==/d' pyproject.toml
echo "Modified pyproject.toml (vllm entry):"
grep -i vllm pyproject.toml || echo "No vllm entry found (expected after removal)"

# Install Prime-RL dependencies using pip install --system
# This follows the same pattern as other integration tests in test-pipeline.yaml
echo ""
echo "=========================================="
echo "=== Installing Prime-RL Dependencies ==="
echo "=========================================="
echo "This may take several minutes..."

# Install Prime-RL in editable mode with its dependencies
# Constrain NumPy to be compatible with vLLM's numba dependency (requires NumPy <= 2.2)
echo "Installing Prime-RL package and dependencies..."
echo "Constraining NumPy version to maintain numba compatibility (numba requires NumPy <= 2.2)"
uv pip install --system -e . --prerelease=allow 'numpy<2.3'

echo ""
echo "Prime-RL installation complete!"
echo "Using PyTorch and vLLM from Docker image (preserves binary compatibility)"

# ==========================================
# Verify Installation
# ==========================================
echo ""
echo "=========================================="
echo "=== Verifying Installations ==="
echo "=========================================="

echo ""
echo "=== Python Version ==="
python3 -c "import sys; print(f'Python: {sys.version}')"

echo ""
echo "=== vLLM Installation ==="
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}'); print(f'vLLM location: {vllm.__file__}')"

echo ""
echo "=== PyTorch Installation ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "=== Prime-RL Installation ==="
python3 -c "import prime_rl; print(f'Prime-RL imported successfully'); print(f'Prime-RL location: {prime_rl.__file__}')"

echo ""
echo "=== Key Dependencies ==="
echo "Checking critical packages:"
python3 -m pip list 2>/dev/null | grep -E "(vllm|torch|flashinfer|transformers|apache-tvm|triton)" || echo "Could not list packages"

echo ""
echo "Prime-RL integration test environment setup complete!"

# ==========================================
# Pre-flight vLLM Server Health Check
# ==========================================
echo ""
echo "=========================================="
echo "=== Pre-flight vLLM Server Test ==="
echo "=========================================="
echo "Starting vLLM server manually to verify it can start..."
echo "This helps diagnose server startup issues before running pytest"

# Use a small model for quick testing
TEST_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
echo "Test model: ${TEST_MODEL}"

# Start server in background with full logging
export VLLM_LOGGING_LEVEL=DEBUG
echo "Starting vLLM server with DEBUG logging..."

vllm serve "${TEST_MODEL}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  2>&1 | tee /tmp/vllm_preflight.log &

VLLM_PID=$!
echo "vLLM server started with PID: ${VLLM_PID}"

# Wait for health check with detailed status
HEALTH_URL="http://localhost:8000/health"
MAX_WAIT=180
elapsed=0
echo "Checking health endpoint: ${HEALTH_URL}"
echo "Maximum wait time: ${MAX_WAIT} seconds"

while [ $elapsed -lt $MAX_WAIT ]; do
  if curl -f -s "${HEALTH_URL}" > /dev/null 2>&1; then
    echo "✓ vLLM server is healthy after ${elapsed}s"
    
    # Server is healthy, now shut it down gracefully
    echo "Shutting down pre-flight vLLM server..."
    kill "${VLLM_PID}" 2>/dev/null || true
    
    # Wait a bit for graceful shutdown
    sleep 2
    
    # Force kill if still running
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "Force killing pre-flight server..."
      kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
    
    wait "${VLLM_PID}" 2>/dev/null || true
    echo "Pre-flight vLLM server shutdown complete"
    break
  fi
  
  # Check if process is still running
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "✗ vLLM server process died after ${elapsed}s"
    echo ""
    echo "=== Last 100 lines of server log ==="
    tail -100 /tmp/vllm_preflight.log
    echo ""
    echo "=== Full server log (first 500 lines) ==="
    head -500 /tmp/vllm_preflight.log
    exit 1
  fi
  
  # Show progress every 10 seconds
  if [ $((elapsed % 10)) -eq 0 ] && [ $elapsed -gt 0 ]; then
    echo "  Still waiting for server... (${elapsed}/${MAX_WAIT}s)"
    echo "  Process status: $(ps -p ${VLLM_PID} -o stat= 2>/dev/null || echo 'unknown')"
  fi
  
  sleep 5
  elapsed=$((elapsed + 5))
done

if [ $elapsed -ge $MAX_WAIT ]; then
  echo "✗ vLLM server did not become healthy within ${MAX_WAIT}s"
  echo ""
  echo "=== Full server log ==="
  cat /tmp/vllm_preflight.log
  echo ""
  echo "=== Process information ==="
  ps -p "${VLLM_PID}" -f 2>/dev/null || echo "Process not found"
  kill "${VLLM_PID}" 2>/dev/null || true
  exit 1
fi

echo ""
echo "✓ Pre-flight check passed!"
echo "vLLM server can start successfully"

# Give the system a moment to fully release resources
sleep 3

# ==========================================
# Run Prime-RL Integration Tests
# ==========================================
echo ""
echo "=========================================="
echo "=== Running Prime-RL Integration Tests ==="
echo "=========================================="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""
echo "Test command: pytest -v -s tests/integration/test_rl.py -m gpu"
echo "Test timeout: 600 seconds (10 minutes)"
echo ""
echo "Note: Prime-RL's conftest.py redirects vLLM server logs to DEVNULL"
echo "      If tests fail, check pre-flight diagnostics above"
echo ""

export WANDB_MODE=offline
export VLLM_LOGGING_LEVEL=DEBUG

# Run with timeout and capture output
echo "Starting tests..."
timeout 600 pytest -v -s tests/integration/test_rl.py -m gpu 2>&1 | tee /tmp/pytest_output.log

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "=== Test Execution Summary ==="
echo "=========================================="
echo "Test exit code: ${TEST_EXIT_CODE}"
echo "End timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [ $TEST_EXIT_CODE -eq 124 ]; then
  echo "✗ Tests timed out after 600 seconds"
  TEST_EXIT_CODE=1
fi

if [ $TEST_EXIT_CODE -ne 0 ]; then
  echo ""
  echo "=========================================="
  echo "=== Post-Failure Diagnostics ==="
  echo "=========================================="
  
  # Check for any vLLM processes
  echo ""
  echo "=== Active vLLM processes ==="
  ps aux | grep -i vllm | grep -v grep || echo "No vLLM processes found"
  
  # Check port usage
  echo ""
  echo "=== Port 8000 usage ==="
  netstat -tlnp 2>/dev/null | grep 8000 || ss -tlnp 2>/dev/null | grep 8000 || echo "Port 8000 not in use"
  
  # GPU state
  echo ""
  echo "=== GPU state after failure ==="
  nvidia-smi || echo "nvidia-smi not available"
  
  # System resources after failure
  echo ""
  echo "=== System resources after failure ==="
  echo "Memory:"
  free -h || true
  echo ""
  echo "Disk:"
  df -h . || true
  
  # Last lines of pytest output
  echo ""
  echo "=== Last 50 lines of pytest output ==="
  tail -50 /tmp/pytest_output.log || echo "Could not read pytest output log"
  
  exit $TEST_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "✓ Prime-RL integration tests completed successfully!"
echo "=========================================="
echo "End Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
