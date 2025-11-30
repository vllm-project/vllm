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
echo "Installing Prime-RL using system packages..."
echo "NOTE: Using 'uv pip install --system' to preserve Docker's PyTorch 2.9.0+cu129"
echo "      'uv sync' would downgrade PyTorch to 2.8 and break vLLM binary compatibility"

# Remove vllm pin from pyproject.toml to use pre-installed vLLM
echo ""
echo "Removing vllm pin from pyproject.toml..."
sed -i '/vllm==/d' pyproject.toml
echo "Modified pyproject.toml (vllm entry removed)"

# Install Prime-RL dependencies
echo ""
echo "=========================================="
echo "=== Installing Prime-RL Dependencies ==="
echo "=========================================="
echo "This may take several minutes..."

# Install Prime-RL with NumPy constraint to prevent incompatible upgrades
# Use --system to install into Docker's Python (preserves PyTorch 2.9)
echo "Installing Prime-RL with NumPy<2.3 constraint (for numba compatibility)..."
uv pip install --system -e . --prerelease=allow 'numpy<2.3'

echo ""
echo "Prime-RL installation complete!"
echo "Preserved Docker's PyTorch and vLLM to maintain binary compatibility"

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
# Patch Prime-RL conftest.py for debugging
# ==========================================
echo ""
echo "=========================================="
echo "=== Patching Prime-RL for Diagnostics ==="
echo "=========================================="
echo "Modifying conftest.py to capture vLLM server logs instead of redirecting to DEVNULL"

CONFTEST_PATH="tests/conftest.py"
if [ -f "${CONFTEST_PATH}" ]; then
  echo "Found conftest.py at: ${CONFTEST_PATH}"
  echo "Original conftest.py size: $(wc -l < ${CONFTEST_PATH}) lines"
  
  # Backup original
  cp "${CONFTEST_PATH}" "${CONFTEST_PATH}.backup"
  
  # Show the original Popen line before patching
  echo ""
  echo "Original Popen line:"
  grep -n "subprocess.Popen.*VLLM_SERVER_CMD" "${CONFTEST_PATH}" || echo "Pattern not found"
  
  # Replace stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL with file outputs
  # Also replace just stderr=subprocess.DEVNULL in case stdout is separate
  sed -i 's/stdout=subprocess\.DEVNULL, stderr=subprocess\.DEVNULL/stdout=open("\/tmp\/vllm_server_stdout.log", "w"), stderr=open("\/tmp\/vllm_server_stderr.log", "w")/g' "${CONFTEST_PATH}"
  sed -i 's/stderr=subprocess\.DEVNULL/stderr=open("\/tmp\/vllm_server_stderr.log", "w")/g' "${CONFTEST_PATH}"
  sed -i 's/stdout=subprocess\.DEVNULL/stdout=open("\/tmp\/vllm_server_stdout.log", "w")/g' "${CONFTEST_PATH}"
  
  echo ""
  echo "After patching - Popen line:"
  grep -n "subprocess.Popen.*VLLM_SERVER_CMD" "${CONFTEST_PATH}" || echo "Pattern not found"
  
  echo ""
  echo "Patched conftest.py to log server output to:"
  echo "  - /tmp/vllm_server_stdout.log"
  echo "  - /tmp/vllm_server_stderr.log"
  
  # Pre-create log files to ensure they're writable
  touch /tmp/vllm_server_stdout.log /tmp/vllm_server_stderr.log
  chmod 666 /tmp/vllm_server_stdout.log /tmp/vllm_server_stderr.log
  echo "Pre-created log files:"
  ls -lh /tmp/vllm_server_*.log
  
  # Show what command will be used
  echo ""
  echo "Extracting vLLM server configuration from conftest.py:"
  echo ""
  grep -B 2 -A 8 "VLLM_SERVER_CMD\|VLLM_SERVER_ENV" "${CONFTEST_PATH}" | head -40 || echo "Not found in conftest.py"
  
  # Patch VLLM_SERVER_CMD to use system Python instead of 'uv run'
  echo ""
  echo "=== Patching VLLM_SERVER_CMD for system Python ==="
  echo "Original command uses 'uv run' which requires UV environment"
  echo "Changing to use system Python directly since we installed with --system"
  
  # Replace ["uv", "run", "inference", ... with ["python3", "-m", "prime_rl.inference.server", ...
  # or just ["inference", ... since it's installed in system PATH
  sed -i 's/VLLM_SERVER_CMD = \["uv", "run", "inference"/VLLM_SERVER_CMD = ["inference"/' "${CONFTEST_PATH}"
  
  echo ""
  echo "Patched VLLM_SERVER_CMD:"
  grep "VLLM_SERVER_CMD" "${CONFTEST_PATH}"
  
  # Test that the inference command exists
  echo ""
  echo "=== Testing Prime-RL inference command ==="
  if inference --help > /tmp/inference_help.log 2>&1; then
    echo "✓ 'inference' command is available in system PATH"
    head -20 /tmp/inference_help.log
  else
    echo "✗ 'inference' command failed"
    cat /tmp/inference_help.log
  fi
else
  echo "Warning: ${CONFTEST_PATH} not found"
fi

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
echo "      Server startup diagnostics will be limited - check installation verification above"
echo ""

export WANDB_MODE=offline
export VLLM_LOGGING_LEVEL=DEBUG

echo "Environment variables for test run:"
echo "  WANDB_MODE=${WANDB_MODE}"
echo "  VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}"
echo ""

# Start a background monitor to show log file growth
echo "Starting background log monitor..."
(
  while true; do
    sleep 10
    if [ -f "/tmp/vllm_server_stdout.log" ] || [ -f "/tmp/vllm_server_stderr.log" ]; then
      echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Log file status:"
      ls -lh /tmp/vllm_server_*.log 2>/dev/null || true
      if [ -f "/tmp/vllm_server_stderr.log" ] && [ -s "/tmp/vllm_server_stderr.log" ]; then
        echo "  [STDERR LAST 5 LINES]:"
        tail -5 /tmp/vllm_server_stderr.log | sed 's/^/    /'
      fi
    fi
  done
) &
MONITOR_PID=$!
echo "Background monitor started with PID: ${MONITOR_PID}"

# Run with timeout and capture output
echo ""
echo "Starting tests..."
set +e  # Temporarily disable exit on error to capture exit code
timeout 600 pytest -v -s tests/integration/test_rl.py -m gpu 2>&1 | tee /tmp/pytest_output.log
TEST_EXIT_CODE=${PIPESTATUS[0]}
set -e  # Re-enable exit on error

# Stop background monitor
kill ${MONITOR_PID} 2>/dev/null || true
echo ""
echo "Background monitor stopped"

# Immediately check log files after test completion
echo ""
echo "=== Immediate Post-Test Log Check ==="
echo "Checking if vLLM server logs were created:"
ls -lh /tmp/vllm_server_*.log 2>/dev/null || echo "No log files found"
echo ""
echo "All files in /tmp:"
ls -lh /tmp/ | head -50

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
  
  # Check for log files first
  echo ""
  echo "=== Checking ALL /tmp files ==="
  ls -lh /tmp/ | grep -E "vllm|pytest|inference" || echo "No relevant files in /tmp"
  
  echo ""
  echo "=== Checking for vLLM server log files specifically ==="
  ls -lh /tmp/vllm_server_*.log 2>/dev/null || echo "No vLLM server log files found in /tmp"
  
  # Show vLLM server logs if they exist (from our conftest.py patch)
  echo ""
  echo "=========================================="
  echo "=== vLLM Server Logs (from pytest fixture) ==="
  echo "=========================================="
  
  if [ -f "/tmp/vllm_server_stdout.log" ]; then
    STDOUT_SIZE=$(wc -l < /tmp/vllm_server_stdout.log)
    echo ""
    echo "Server stdout (${STDOUT_SIZE} lines):"
    echo "----------------------------------------"
    cat /tmp/vllm_server_stdout.log
    echo "----------------------------------------"
  else
    echo "✗ No stdout log found at /tmp/vllm_server_stdout.log"
  fi
  
  echo ""
  if [ -f "/tmp/vllm_server_stderr.log" ]; then
    STDERR_SIZE=$(wc -l < /tmp/vllm_server_stderr.log)
    echo ""
    echo "Server stderr (${STDERR_SIZE} lines):"
    echo "----------------------------------------"
    cat /tmp/vllm_server_stderr.log
    echo "----------------------------------------"
  else
    echo "✗ No stderr log found at /tmp/vllm_server_stderr.log"
  fi
  
  # Check for any vLLM processes
  echo ""
  echo "=== Active vLLM/inference processes ==="
  ps aux | grep -E "vllm|inference" | grep -v grep || echo "No vLLM/inference processes found"
  
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
  echo "=== Last 100 lines of pytest output ==="
  tail -100 /tmp/pytest_output.log || echo "Could not read pytest output log"
  
  exit $TEST_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "✓ Prime-RL integration tests completed successfully!"
echo "=========================================="
echo "End Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
