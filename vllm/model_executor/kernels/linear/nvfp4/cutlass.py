# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import (
    cutlass_scaled_fp4_mm,
    scaled_fp4_quant,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class CutlassNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the vLLM CUTLASS kernel."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if cutlass_fp4_supported():
            return True, None
        return False, "CUTLASS FP4 kernels not available"

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
        *,
        input_global_scale_inv: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_global_scale_inv = (
            layer.input_global_scale_inv
            if input_global_scale_inv is None
            else input_global_scale_inv
        )
        alpha = layer.alpha if alpha is None else alpha
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)

        # input_global_scale_inv is the FP4 quant multiplier (2688 / amax):
        # scalar for the per-tensor path, length-M for the per-token path.
        # Per-token is required for CUDA-graph correctness (a per-tensor amax
        # reduces over the padded graph buffer, and padding rows corrupt real
        # rows' scale). The GEMM epilogue only takes a scalar alpha, so for the
        # per-token path we pre-scale each row, quantize with a unit global
        # scale (each row's block scale then comes from its own amax), and apply
        # the per-row factor as a post-GEMM row multiply -- identical to a
        # per-row epilogue alpha and CUDA-graph-safe (elementwise on graph
        # tensors). A fused per-row epilogue is future CUTLASS work.
        per_row = input_global_scale_inv.numel() != 1
        if per_row:
            quant_input = x * input_global_scale_inv.reshape(*x.shape[:-1], 1).to(
                x.dtype
            )
            quant_global_scale = x.new_ones((), dtype=torch.float32)
            # alpha[i] = input_global_scale[i] * weight_global_scale; the shared
            # weight factor goes to the GEMM, the per-row part is applied after.
            gemm_alpha = layer.weight_global_scale.reshape(()).to(torch.float32)
        else:
            quant_input = x
            quant_global_scale = input_global_scale_inv
            gemm_alpha = alpha

        x_fp4, x_blockscale = scaled_fp4_quant(
            quant_input,
            quant_global_scale,
            is_sf_swizzled_layout=True,
            backend="cutlass",
            padded_n=x.shape[-1] + weights_padding_bytes * 2,
        )

        out = cutlass_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            gemm_alpha,
            output_dtype,
        )

        if per_row:
            # Recover the per-row input scale (GEMM already applied the shared
            # weight factor) and multiply each output row by it.
            per_row_scale = (alpha / gemm_alpha).reshape(-1, 1)
            out = (out.to(torch.float32) * per_row_scale).to(output_dtype)

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
