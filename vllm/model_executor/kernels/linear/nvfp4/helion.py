# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
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
# TODO: for now i am importing it like any other external repo(flash-infer), the todo is to port it to helion repo!
#      -- but we will need to decided /or if to avoid the customop
sys.path.insert(0, '/home/redhat-et/src/lkesem/helion/examples')
from nvfp4_gemv import nvfp4_gemv_fp4in
from vllm._custom_ops import scaled_fp4_quant

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
        
        if x_fp4.shape[0] != 1:
            #print(f"CUTLASS path: M={x_fp4.shape[0]},N={layer.weight.shape[0]},K={ x_fp4.shape[1]}") # working with eager mode!
            out = cutlass_scaled_fp4_mm(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                output_dtype,
            )
        else:
            # use gemv_helion
            k_bytes = layer.weight.shape[1]
            backend = "cute" if k_bytes % 2048 == 0 else "triton"
            #print(f" gemv_helion path: M={x_fp4.shape[0]},N={layer.weight.shape[0]},K={ x_fp4.shape[1]} with backend {backend}")

            alpha_float = float(layer.alpha)
            out = nvfp4_gemv_fp4in(layer.weight, x_fp4.flatten(), layer.weight_scale,
                                   x_blockscale, alpha=alpha_float, backend=backend).unsqueeze(0)

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
