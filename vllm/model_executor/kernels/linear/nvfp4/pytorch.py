# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class TorchNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM implemented with PyTorch's native ``torch._scaled_mm``.

    Eager execution uses PyTorch's native scaled-matmul implementation. Under
    ``torch.compile``, TorchInductor may select any enabled scaled-GEMM backend,
    including NVGEMM on supported Blackwell systems.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_device_capability_family(100):
            return False, "Torch NVFP4 requires sm_10x (Blackwell)"
        if not hasattr(torch, "_scaled_mm"):
            return False, "torch._scaled_mm not available"
        if not hasattr(torch, "float4_e2m1fn_x2"):
            return False, "torch.float4_e2m1fn_x2 not available"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="cutlass",
            padded_n=x.shape[-1] + weights_padding_bytes * 2,
        )

        # scaled_fp4_quant returns packed bytes, while torch._scaled_mm uses
        # the corresponding shell dtype to describe the two FP4 values stored
        # in each byte.
        x_fp4 = x_fp4.view(torch.float4_e2m1fn_x2)
        weight = layer.weight
        if weight.dtype != torch.float4_e2m1fn_x2:
            weight = weight.view(torch.float4_e2m1fn_x2)

        out = torch._scaled_mm(
            x_fp4,
            weight.t(),
            scale_a=x_blockscale.reshape(-1),
            scale_b=layer.weight_scale.reshape(-1),
            out_dtype=output_dtype,
        )

        # ModelOpt stores the product of the activation and weight global
        # dequantization scales as alpha. TorchInductor can fold this multiply
        # into an NVGEMM output-scale argument.
        out = out * layer.alpha
        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
