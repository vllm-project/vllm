# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453


def maybe_launch_pdl(value: bool = False) -> dict:
    """Return launch metadata for Triton kernel calls that may use PDL.

    The ``launch_pdl`` launch attribute (Programmatic Dependent Launch) is
    a NVIDIA Hopper SM90+ feature exposed by NVIDIA's Triton runtime.
    Other Triton backends (notably ROCm/HIP) do not recognize this kwarg
    and raise ``KeyError`` from ``JITKernel._pack_args``. Use this helper
    in the kernel call site:

        kernel[grid](..., **maybe_launch_pdl())

    so the attribute is only forwarded on platforms whose Triton runtime
    supports it.
    """
    # Lazy import to avoid pulling in the full platform stack at module
    # import time of vllm.triton_utils.
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        return {"launch_pdl": value}
    return {}


__all__ = [
    "HAS_TRITON",
    "triton",
    "tl",
    "tldevice",
    "LOG2E",
    "LOGE2",
    "maybe_launch_pdl",
]
