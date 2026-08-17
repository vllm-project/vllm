# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    kE2M1ToFloat_handle,
    run_nvfp4_emulations,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class EmulationNvFp4LinearKernel(NvFp4LinearKernel):
    """Software emulation fallback for NVFP4 (dequant → BF16 matmul)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Always available as a last-resort fallback.
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Move the E2M1 lookup table to the device now, because
        # `.to(device)` is not allowed during CUDA graph capture.
        kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(layer.weight.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = run_nvfp4_emulations(
            x=x,
            input_global_scale=layer.input_global_scale_inv,
            weight=layer.weight,
            weight_scale_swizzled=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            swizzle=False,
        )
        if bias is not None:
            out = out + bias
        return out
