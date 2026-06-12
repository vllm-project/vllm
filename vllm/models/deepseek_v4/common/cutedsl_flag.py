# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime gate for CUTeDSL kernel dispatch in DeepSeek V4 common ops.

The dispatchers in ``common/ops/`` (e.g. ``dequantize_and_gather_k_cache``,
``fused_indexer_q_rope_quant``) and the ``DeepseekCompressor`` opportunistically
use CUTeDSL kernels from ``nvidia/ops/`` when the optional ``cutlass`` package
is importable.

The hw-agnostic code path does not dispatch to those NVIDIA-only kernels
"""

import vllm.envs as envs
from vllm.utils.import_utils import has_cutedsl


def cutedsl_enabled() -> bool:
    """Return True if CUTeDSL kernels should be used.

    True only when (a) the hw-agnostic stream is not active and (b) the
    ``cutlass`` package is importable.
    """
    if envs.VLLM_USE_HW_AGNOSTIC:
        return False
    return has_cutedsl()
