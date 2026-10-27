# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 model — hardware-isolated entry point.

The implementation lives under ``nvidia/`` and ``amd/``; this module picks the
right one for the current platform and re-exports the public classes used by
the model registry. (Mirrors ``vllm.models.deepseek_v4`` / ``qwen4_exp``.)

Helpers such as ``common.rocm_compat`` must be importable without loading the
NVIDIA or AMD model graphs.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.model import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )
    from .nvidia.mtp import MiniMaxM3MTP

__all__ = [
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from .amd.model import (  # type: ignore[assignment]
            MiniMaxM3SparseForCausalLM,
            MiniMaxM3SparseForConditionalGeneration,
        )
        from .amd.mtp import MiniMaxM3MTP  # type: ignore[assignment]
    else:
        from .nvidia.model import (
            MiniMaxM3SparseForCausalLM,
            MiniMaxM3SparseForConditionalGeneration,
        )
        from .nvidia.mtp import MiniMaxM3MTP

    exported = {
        "MiniMaxM3MTP": MiniMaxM3MTP,
        "MiniMaxM3SparseForCausalLM": MiniMaxM3SparseForCausalLM,
        "MiniMaxM3SparseForConditionalGeneration": (
            MiniMaxM3SparseForConditionalGeneration
        ),
    }
    globals().update(exported)
    return exported[name]
