# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority invariants for NVFP4 linear kernel auto-selection."""

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    MarlinNvFp4LinearKernel,
)
from vllm.platforms import PlatformEnum

_CUDA_KERNELS = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]


def test_w4a16_cutedsl_stays_in_the_a16_tier():
    """Complements ``test_nvfp4_kernel_selection.py``, which pins the W4A16
    kernel below the W4A4 kernels that run on sm_12x. This pins the other
    side: it must stay immediately above Marlin.

    The A16 kernels form a tier -- Marlin has always outranked Trtllm, cuDNN
    and Fbgemm, so "every W4A4 kernel first" is not the invariant here.
    Demoting the CuTe-DSL W4A16 kernel below those three would promote them
    above the A16 tier for the first time on any release; promoting it splits
    the two A16 siblings apart.
    """
    assert (
        _CUDA_KERNELS.index(FlashInferCuteDslNvFp4W4A16LinearKernel)
        == _CUDA_KERNELS.index(MarlinNvFp4LinearKernel) - 1
    ), "the two A16 kernels must stay adjacent, CuTe-DSL W4A16 preferred"
