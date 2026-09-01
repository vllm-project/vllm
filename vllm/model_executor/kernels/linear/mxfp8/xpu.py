# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

_MXFP8_GROUP_SIZE = 32


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

    def __init__(self, config: Mxfp8LinearLayerConfig):
        super().__init__(config)
        self._quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, _MXFP8_GROUP_SIZE),
            use_ue8m0=True,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Checkpoint scale is [N, K//32] (one E8M0 scale per 32 elements).
        # oneDNN fp8_gemm requires contiguous [K//32, N] layout. Store the
        # transposed contiguous buffer as a .t() view so that:
        #   - dequantization consumers still see the checkpoint shape
        #   - apply_weights recovers the oneDNN layout via .t() at zero cost
        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        scale_kn = weight_scale.data.t().contiguous()
        replace_parameter(layer, "weight_scale", scale_kn.t())

        if getattr(layer, "is_bmm", False):
            self._prepare_bmm_params(layer, scale_kn)

    def _prepare_bmm_params(
        self, layer: torch.nn.Module, scale_kn: torch.Tensor
    ) -> None:
        """Precompute batched weight and scale for grouped fp8_bmm (e.g. wo_a).

        Splits scale [K//32, N_total] into [G, K//32, N_per_group] and weight
        [N_total, K] into contiguous [G, K, N_per_group] for batch GEMM.
        """
        batch = layer.bmm_batch_size
        k_blocks, n_blocks = scale_kn.shape
        layer.bmm_scale = (
            scale_kn.reshape(k_blocks, batch, n_blocks // batch)
            .permute(1, 0, 2)
            .contiguous()
        )
        w = layer.weight
        n_total, k = w.shape
        layer.bmm_weight = (
            w.reshape(batch, n_total // batch, k).permute(0, 2, 1).contiguous()
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        x_fp8, x_scale = self._quant(x)
        x_scale = x_scale.to(torch.float8_e8m0fnu)
        return torch.ops._xpu_C.fp8_gemm(
            x_fp8,
            layer.weight.t(),
            out_dtype,
            x_scale,
            layer.weight_scale.t(),
            bias,
        )
