#!/usr/bin/env bash
# Run from any directory in this checkout. Requires uv and a CUDA 13 capable driver.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
uv venv --python 3.12 .deps/vllm
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=b6e7c1f1f0430b5d4784aea391c42067581b7f76 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    uv pip install --python .deps/vllm/bin/python -e . --torch-backend=cu130
uv pip install --python .deps/vllm/bin/python \
    'aiperf==0.12.0' 'flashinfer-python==0.6.18.post1'
uv venv --python 3.12 .deps/sglang
uv pip install --python .deps/sglang/bin/python --torch-backend=cu130 \
    --overrides <(printf '%s\n' 'flashinfer-python==0.6.18.post1' \
        'nvidia-cutlass-dsl[cu13]==4.7.1') \
    'sglang==0.5.17' 'sglang-kernel==0.4.5' \
    'torch==2.11.0' 'torchvision==0.26.0' \
    'aiperf==0.12.0' 'flash-attn-4==4.0.0b19'
uv pip check --python .deps/vllm/bin/python
# SGLang metadata pins older FlashInfer/CUTLASS; the overrides match our capture.
# A plain `uv pip check` reports those two intentional metadata mismatches.
