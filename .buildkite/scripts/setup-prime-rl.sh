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

# Clone Prime-RL repository
echo "Cloning Prime-RL repository..."
git clone "${PRIME_RL_REPO}" "${PRIME_RL_DIR}"
cd "${PRIME_RL_DIR}"

# Sync Prime-RL dependencies
echo "Installing Prime-RL dependencies..."
uv sync && uv sync --all-extras

# Remove vllm pin from pyproject.toml
echo "Removing vllm pin from pyproject.toml..."
sed -i '/vllm==/d' pyproject.toml

# Install nightly vLLM to override the version in Prime-RL
echo "Installing nightly vLLM..."
uv add vllm --index https://wheels.vllm.ai/nightly

# Verify installation
echo "Verifying installations..."
uv run python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
uv run python -c "import prime_rl; print('Prime-RL imported successfully')"

echo "Prime-RL integration test environment setup complete!"
