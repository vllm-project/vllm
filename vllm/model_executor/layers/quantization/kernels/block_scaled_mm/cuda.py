# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.flashinfer import (
    should_use_flashinfer_for_blockscale_fp8_gemm,
)

from .BlockScaledMMKernel import Fp8BlockMMScaledConfig, Fp8BlockScaledMMKernel
from .cutlass import CutlassBlockScaledMMKernel
from .deep_gemm import DeepGemmBlockScaledMMKernel
from .flashinfer import FlashInferDeepGEMMDynamicBlockScaledKernel
from .triton import TritonBlockScaledMMKernel


class CudaBlockScaledMMKernel(Fp8BlockScaledMMKernel):
    def __init__(self, config: Fp8BlockMMScaledConfig) -> None:
        self.flashinfer_deepgemm_kernel: (
            FlashInferDeepGEMMDynamicBlockScaledKernel | None
        ) = None
        if FlashInferDeepGEMMDynamicBlockScaledKernel.is_supported()[0]:
            self.flashinfer_deepgemm_kernel = (
                FlashInferDeepGEMMDynamicBlockScaledKernel(config)
            )
        self.deepgemm_kernel: DeepGemmBlockScaledMMKernel | None = None
        if DeepGemmBlockScaledMMKernel.is_supported()[0]:
            self.deepgemm_kernel = DeepGemmBlockScaledMMKernel(config)
        self.default_fallback_kernel: Fp8BlockScaledMMKernel | None = None
        for kernel in self.ordered_fallback_kernels():
            if kernel.is_supported()[0]:
                self.default_fallback_kernel = kernel(config)

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        if self.deepgemm_kernel is not None:
            self.deepgemm_kernel.process_weights_after_loading(layer)

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type["Fp8BlockScaledMMKernel"]]:
        return [CutlassBlockScaledMMKernel, TritonBlockScaledMMKernel]

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "only cuda devices are supported."
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        weight = params.weight

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_dtype = x.dtype

        if (
            self.flashinfer_deepgemm_kernel is not None
            and should_use_flashinfer_for_blockscale_fp8_gemm(
                True, output_dtype, input_2d, weight
            )
            and should_use_deepgemm_for_fp8_linear(output_dtype, weight, True)
        ):
            return self.flashinfer_deepgemm_kernel.apply_weights(layer, x, bias)

        if self.deepgemm_kernel is not None and should_use_deepgemm_for_fp8_linear(
            output_dtype, weight, True
        ):
            return self.deepgemm_kernel.apply_weights(layer, bias)

        assert self.default_fallback_kernel is not None
        return self.default_fallback_kernel.apply_weights(layer, x, bias)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert self.default_fallback_kernel is not None
        return self.default_fallback_kernel.apply_block_scaled_mm(
            A, B, out_dtype, As, Bs, **kwargs
        )
