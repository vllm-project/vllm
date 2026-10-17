#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One-command entry point for the CC bridge microbench.
#   bash benchmarks/confidential_compute/run.sh            # full suite (~5-10 min)
#   bash benchmarks/confidential_compute/run.sh --quick    # smoke run (~1 min)
# Extra arguments are passed through to cc_bridge_bench.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "error: '$PYTHON' needs PyTorch with a working CUDA GPU." >&2
    echo "       pip install -r $HERE/requirements.txt" >&2
    exit 1
fi
if ! "$PYTHON" -c "import pynvml" 2>/dev/null; then
    echo "warning: nvidia-ml-py is missing, so the CC state cannot be detected;" >&2
    echo "         install it or pass --cc-label on|off." >&2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi conf-compute -f 2>/dev/null || true
fi

exec "$PYTHON" "$HERE/cc_bridge_bench.py" "$@"
