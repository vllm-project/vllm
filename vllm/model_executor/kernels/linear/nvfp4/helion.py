# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

import torch

from vllm._custom_ops import (
    scaled_fp4_quant,
)
from vllm.kernels.helion.ops.nvfp4_gemm import (
    nvfp4_gemm_w4a4,
)
from vllm.kernels.helion.ops.nvfp4_gemv import (
    nvfp4_gemv_fp4in,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = logging.getLogger(__name__)

def _nvfp4_gemv_fp4in_impl(
    output: torch.Tensor,
    weight: torch.Tensor,
    x_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    x_blockscale: torch.Tensor,
    alpha: torch.Tensor,
    alpha_scalar: float,
) -> None:
    """Dispatch between CUTLASS (M>1) and Helion GEMV (M=1) for NVFP4."""
    M = x_fp4.shape[0]
    # make scales M-major for blockwise quant, doesn't affect 1D scales
    if M != 1:
        # CUTLASS path for batch size > 1
        weight_t = weight.view(torch.uint8).t().contiguous()
        result = nvfp4_gemm_w4a4(
            x_fp4,
            weight_t,  # use weight_t, not weight
            x_blockscale,
            weight_scale,
            alpha=alpha_scalar,  # use alpha_scalar (float), not alpha (Tensor)
        )  # no 6th arg
        if result.dtype != output.dtype:
            result = result.to(output.dtype)
        output.copy_(result)

    # Helion GEMV path for batch size == 1
    else:
        # logger.debug(f"[HELION GEMV] M={M}, N={weight.shape[0]}, K={x_fp4.shape[1]}")
        # weight_scale = weight_scale.t().contiguous().t()
        # make scales K-major for blockwise quant, doesn't affect 1D scales
        # x_blockscale = x_blockscale.t().contiguous().t()
        result = nvfp4_gemv_fp4in(
            weight,
            x_fp4,  # .flatten(),
            weight_scale,
            x_blockscale,
            alpha=alpha_scalar,
        )
        # if output.dim() == 1:
        output.copy_(result)  # 1D → 1D
        # else:
        #    output.copy_(result.view(1, -1))  # Reshape for 2D output


def _nvfp4_gemv_fp4in_fake(
    output: torch.Tensor,
    weight: torch.Tensor,
    x_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    x_blockscale: torch.Tensor,
    alpha: torch.Tensor,
    alpha_scalar: float,
) -> None:
    return


direct_register_custom_op(
    "nvfp4_gemv_fp4in",
    _nvfp4_gemv_fp4in_impl,
    mutates_args=["output"],
    fake_impl=_nvfp4_gemv_fp4in_fake,
)


class HelionNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the vLLM Helion kernel."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if cutlass_fp4_supported():
            return True, None
        return False, "Helion FP4 kernels not available"

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
        layer.alpha_scalar = float(layer.alpha.item())
        layer.weight_gemm = torch.nn.Parameter(
            padded_weight.view(torch.uint8).t().contiguous(),
            requires_grad=False,
        )

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
        M = x_fp4.shape[0]
        if M == 1:
            out = torch.empty(
                (x_fp4.shape[0], layer.weight.shape[0]),
                dtype=output_dtype,
                device=x.device,
            )
            torch.ops.vllm.nvfp4_gemv_fp4in(
                out,
                layer.weight,
                x_fp4,
                layer.weight_scale,
                x_blockscale,
                alpha=layer.alpha,
                alpha_scalar=layer.alpha_scalar,
            )
        else:
            out = nvfp4_gemm_w4a4(
                x_fp4,  # A_packed [M, K//2]
                layer.weight_gemm,  # B_packed [K//2, N] (pre-transposed)
                x_blockscale,  # act_scale
                layer.weight_scale,  # weight_scale
                alpha=layer.alpha_scalar,
            )
            if out.dtype != output_dtype:
                out = out.to(output_dtype)

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
