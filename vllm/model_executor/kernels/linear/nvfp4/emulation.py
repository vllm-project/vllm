# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
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


class EmulationA16NvFp4LinearKernel(NvFp4LinearKernel):
    """Software emulation fallback for W4A16 NVFP4 linear layers.

    Unlike W4A4 NVFP4, W4A16 keeps activations in BF16/FP16, so this path
    dequantizes only the packed NVFP4 weights and runs a plain matmul. It is
    a correctness fallback for builds/platforms where the Marlin FP4 kernels
    are unavailable; it makes no performance claims.
    """

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
        output_shape = [*x.shape[:-1], layer.output_size_per_partition]
        x_2d = x.reshape(-1, x.shape[-1])
        weight = dequantize_to_dtype(
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            x.dtype,
            block_size=16,
            swizzle=False,
        )
        out = torch.matmul(x_2d, weight.t())
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
