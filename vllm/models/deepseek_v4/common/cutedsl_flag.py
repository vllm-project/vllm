# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime flag controlling CUTeDSL kernel dispatch in DeepSeek V4 common ops.

The dispatchers in ``common/ops/`` (e.g. ``dequantize_and_gather_k_cache``,
``fused_indexer_q_rope_quant``) and the ``DeepseekCompressor`` opportunistically
use CUTeDSL kernels from ``nvidia/ops/`` when the optional ``cutlass`` package
is importable. The reference NVIDIA / AMD model paths leave this flag enabled
(default ``True``) so they pick up the CUTeDSL fast paths. The hw-agnostic
model path flips the flag to ``False`` at import time (see
``hw_agnostic/__init__.py``) so the dispatchers always fall back to the
Triton implementations regardless of whether ``cutlass`` happens to be
installed in the environment.
"""

from vllm.utils.import_utils import has_cutedsl

_CUTEDSL_ENABLED = True


def set_cutedsl_enabled(value: bool) -> None:
    global _CUTEDSL_ENABLED
    _CUTEDSL_ENABLED = value


def cutedsl_enabled() -> bool:
    """Return True if CUTeDSL kernels should be used.

    True only when both the runtime flag is on (default; ``False`` on the
    hw-agnostic path) and the ``cutlass`` package is importable.
    """
    return _CUTEDSL_ENABLED and has_cutedsl()
