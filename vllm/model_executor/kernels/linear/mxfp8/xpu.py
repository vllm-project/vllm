# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as quant_mxfp8,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class XPUMxFp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM on XPU."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUMxFp8 only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Transpose scale from checkpoint [N, K//32] to oneDNN [K//32, N]
        # at load time (one-time cost, eliminates per-call .t().contiguous()).
        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        scale_t = weight_scale.data.t().contiguous()
        replace_parameter(layer, "weight_scale", scale_t)

        # For BMM layers (e.g. wo_a), precompute 3D scale and weight:
        if getattr(layer, "is_bmm", False):
            batch = layer.bmm_batch_size
            k_blocks = scale_t.shape[0]  # K//32
            n_per_batch_blocks = scale_t.shape[1] // batch
            layer.bmm_scale = (
                scale_t.reshape(k_blocks, batch, n_per_batch_blocks)
                .permute(1, 0, 2)
                .contiguous()
            )  # [G, K//32, N_per_group]
            # Precompute contiguous [G, K, N] weight for fp8_bmm.
            w = layer.weight.data
            N_total, K = w.shape
            N_per_group = N_total // batch
            layer.bmm_weight = (
                w.reshape(batch, N_per_group, K).permute(0, 2, 1).contiguous()
            )  # [G, K, N]

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        x_fp8, x_scale = quant_mxfp8(x)
        # Weight is [N, K]. Use .t() to create a [K, N] view.
        # Scale is already [K//32, N] from process_weights_after_loading.
        return torch.ops._xpu_C.fp8_gemm(
            x_fp8,
            layer.weight.t(),
            out_dtype,
            x_scale,
            layer.weight_scale,
            bias,
        )
