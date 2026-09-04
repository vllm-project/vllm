# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority invariants for NVFP4 linear kernel auto-selection."""

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    CutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.platforms import PlatformEnum

_CUDA_KERNELS = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]


def test_w4a16_cutedsl_ranks_below_native_w4a4_kernels():
    """``init_nvfp4_linear_kernel`` is first-match-wins, and the CuTe-DSL
    W4A16 kernel accepts every sm_12x device while dequantizing to 16-bit
    activations. Ranked above the native W4A4 kernels it silently captures
    W4A4 checkpoints on sm_12x, where the sm_10x-only CuTe-DSL W4A4 kernel
    above it is rejected -- a ~30% prefill regression on GB10.
    """
    a16_index = _CUDA_KERNELS.index(FlashInferCuteDslNvFp4W4A16LinearKernel)
    for w4a4_kernel in (
        FlashInferCutlassNvFp4LinearKernel,
        FlashInferB12xNvFp4LinearKernel,
        CutlassNvFp4LinearKernel,
    ):
        assert _CUDA_KERNELS.index(w4a4_kernel) < a16_index, (
            f"{w4a4_kernel.__name__} does the W4A4 GEMM natively and must be "
            "tried before the dequantizing W4A16 kernel"
        )
