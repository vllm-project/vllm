#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Prepare the environment for building/testing against PyTorch nightly.
#
# When TORCH_NIGHTLY=1 is set, this script:
#   1. Installs torch, torchvision, torchaudio (unpinned) from the nightly index
#   2. Strips pinned torch versions from requirement files and pyproject.toml
#      using use_existing_torch.py (so they don't conflict with nightly)
#   3. Rewrites --extra-index-url lines in requirement files from stable to nightly
#
# This mirrors what docker/Dockerfile already does for PYTORCH_NIGHTLY=1 builds.
#
# Usage:
#   source .buildkite/scripts/prepare-torch-nightly.sh
#   # or
#   bash .buildkite/scripts/prepare-torch-nightly.sh
#
# The script is a no-op when TORCH_NIGHTLY is not set to "1".

set -euo pipefail

if [ "${TORCH_NIGHTLY:-0}" != "1" ]; then
    echo ">>> TORCH_NIGHTLY is not set, skipping nightly preparation"
    exit 0
fi

echo ">>> Preparing environment for PyTorch nightly..."

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Detect CUDA version for nightly index URL
if command -v nvidia-smi &>/dev/null; then
    CUDA_MAJOR_MINOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    # Use python to get the actual CUDA version from torch if available, else default to cu130
    CUDA_TAG="cu130"
    if python3 -c "import torch" 2>/dev/null; then
        CUDA_TAG=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "cu130")
    fi
    NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/${CUDA_TAG}"
else
    NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cpu"
fi

echo ">>> Using nightly index: ${NIGHTLY_INDEX}"

# Step 1: Install torch nightly (unpinned, --pre)
echo ">>> Installing PyTorch nightly (unpinned)..."
pip install --pre torch torchvision torchaudio --index-url "${NIGHTLY_INDEX}"

echo ">>> Installed torch version: $(python3 -c 'import torch; print(torch.__version__)')"

# Step 2: Strip torch pins from requirements and pyproject.toml
echo ">>> Stripping pinned torch versions from requirement files..."
python3 "${REPO_ROOT}/use_existing_torch.py" --prefix

# Step 3: Rewrite index URLs from stable to nightly
echo ">>> Rewriting index URLs to nightly..."
find "${REPO_ROOT}/requirements" -type f \( -name "*.txt" -o -name "*.in" \) | while read -r reqfile; do
    if grep -q "download.pytorch.org/whl/cu" "$reqfile" 2>/dev/null; then
        sed -i 's|download.pytorch.org/whl/cu|download.pytorch.org/whl/nightly/cu|g' "$reqfile"
        echo "    Updated CUDA index in: ${reqfile#${REPO_ROOT}/}"
    fi
    if grep -q "download.pytorch.org/whl/cpu" "$reqfile" 2>/dev/null; then
        sed -i 's|download.pytorch.org/whl/cpu|download.pytorch.org/whl/nightly/cpu|g' "$reqfile"
        echo "    Updated CPU index in: ${reqfile#${REPO_ROOT}/}"
    fi
done

echo ">>> PyTorch nightly preparation complete"
