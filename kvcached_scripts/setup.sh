#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed"
        echo "Please install uv first, e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        exit 1
    fi
}

setup_vllm() {
    pushd "$ENGINE_DIR"

    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    uv pip install kvcached --no-build-isolation --no-cache-dir

    deactivate
    popd
}

# Check for uv before proceeding
check_uv
setup_vllm
