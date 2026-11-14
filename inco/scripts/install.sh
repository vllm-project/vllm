#!/usr/bin/env bash
# Provision a GPU host for the baseline: vLLM (this fork) + aiperf.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
# shellcheck source=/dev/null
source .venv/bin/activate

# Precompiled kernels: fast editable install of this fork for the baseline run.
# Switch to a source build once you start touching csrc/ or CMakeLists.txt.
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r inco/requirements.txt

.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
aiperf --version
