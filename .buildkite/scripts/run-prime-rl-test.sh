#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Setup script for Prime-RL integration tests
# This script prepares the environment for running Prime-RL tests with nightly vLLM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PRIME_RL_REPO="https://github.com/PrimeIntellect-ai/prime-rl.git"
PRIME_RL_DIR="${REPO_ROOT}/prime-rl"

if command -v rocm-smi &> /dev/null || command -v rocminfo &> /dev/null; then
    echo "AMD GPU detected. Prime-RL currently only supports NVIDIA. Skipping..."
    exit 0
fi

echo "Setting up Prime-RL integration test environment..."

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
fi

# Clone Prime-RL repository at specific branch for reproducible tests
PRIME_RL_BRANCH="integ-vllm-main"
echo "Cloning Prime-RL repository at branch: ${PRIME_RL_BRANCH}..."
git clone --branch "${PRIME_RL_BRANCH}" --single-branch "${PRIME_RL_REPO}" "${PRIME_RL_DIR}"
cd "${PRIME_RL_DIR}"

echo "Setting up UV project environment..."
export UV_PROJECT_ENVIRONMENT=/usr/local
ln -s /usr/bin/python3 /usr/local/bin/python

# Remove vllm pin from pyproject.toml
echo "Removing vllm pin from pyproject.toml..."
sed -i '/vllm==/d' pyproject.toml

# Sync Prime-RL dependencies
echo "Installing Prime-RL dependencies..."
uv sync --inexact && uv sync --inexact --all-extras

# override torch installation, make sure right version is installed
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/test/cu129 --force-reinstall

# Verify installation
echo "Verifying installations..."
uv run python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
uv run python -c "import prime_rl; print('Prime-RL imported successfully')"

echo "Prime-RL integration test environment setup complete!"

echo "Running Prime-RL integration tests..."
export WANDB_MODE=offline # this makes this test not require a WANDB_API_KEY
uv run pytest -vs tests/integration/test_rl.py -m gpu

echo "Prime-RL integration tests completed!"
