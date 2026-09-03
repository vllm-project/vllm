# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 linear kernel selection order (CPU-only).

Run `pytest tests/kernels/quantization/test_nvfp4_kernel_selection.py`.
"""

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    MarlinNvFp4LinearKernel,
)
from vllm.platforms.interface import PlatformEnum

# Kernels that consume BF16 activations. They are correct for weight-only
# checkpoints but leave the FP4 tensor cores idle on a W4A4-capable one.
WEIGHT_ONLY_KERNELS = (
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    MarlinNvFp4LinearKernel,
)


def test_weight_only_kernels_are_not_preferred_over_w4a4():
    """A weight-only kernel must never outrank a W4A4 kernel.

    Selection walks this list and takes the first kernel that reports support
    on the current device. A weight-only kernel placed above a W4A4 one is
    therefore picked on any platform where the kernels above it are gated off,
    even though the checkpoint can feed FP4 activations.
    """
    candidates = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]
    first_weight_only = min(
        candidates.index(kernel)
        for kernel in WEIGHT_ONLY_KERNELS
        if kernel in candidates
    )
    trailing = candidates[first_weight_only:]
    offenders = [k.__name__ for k in trailing if k not in WEIGHT_ONLY_KERNELS]
    assert not offenders, (
        "W4A4 kernels must precede weight-only kernels, but these follow the "
        f"first weight-only entry: {offenders}"
    )
