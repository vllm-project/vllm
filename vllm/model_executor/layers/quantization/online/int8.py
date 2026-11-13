# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    convert_to_int8_moe_kernel_format,
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.online.lm_head import (
    OnlineLMHeadMethodBase,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_weight_quant,
    kInt8DynamicTokenSym,
    kInt8Static32Sym,
    kInt8StaticChannelSym,
    pack_quantized_values_into_int32,
    weight_amax,
)
from vllm.model_executor.utils import replace_parameter
from vllm.scalar_type import scalar_types

# -128 is left unused so the code range is symmetric around 0.
_INT8_MAX = 127
# Codes pack 4 input channels per int32 word.
_PACK_FACTOR = 4


class Int8OnlineMoEMethod(OnlineMoEMethodBase):
    """Online per-channel INT8 MoE quantization.
    Loads fp16/bf16 weights and quantizes them per-row to int8 during loading.
    """

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13_weight, w2_weight = self.get_weights_for_quantization(layer)
        self._quantize_weights(layer, w13_weight, w2_weight)
        self._setup_kernel(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(
        self,
        layer: Module,
        w13_weight: torch.Tensor | None = None,
        w2_weight: torch.Tensor | None = None,
    ) -> None:
        w13_weight = layer.w13_weight if w13_weight is None else w13_weight
        w2_weight = layer.w2_weight if w2_weight is None else w2_weight
        vmax = torch.iinfo(torch.int8).max

        w13 = torch.empty_like(w13_weight, dtype=torch.int8)
        w2 = torch.empty_like(w2_weight, dtype=torch.int8)
        w13_scale = torch.zeros(
            layer.num_experts,
            w13_weight.shape[1],
            device=w13.device,
            dtype=torch.float32,
        )
        w2_scale = torch.zeros(
            layer.num_experts,
            w2_weight.shape[1],
            device=w2.device,
            dtype=torch.float32,
        )

        w2_amax = weight_amax(w2_weight, dim=-1)
        w2_amax = amax_for_moe_weight_quant(w2_amax, self.moe.tp_size)

        for expert in range(layer.local_num_experts):
            # w13: per-row quantization over hidden_size dim
            w = w13_weight[expert, :, :]
            scales = w.abs().amax(dim=1) / vmax
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            w13[expert, :, :] = q.to(torch.int8)
            w13_scale[expert, :] = scales

            # w2: per-row quantization over intermediate_size dim
            w = w2_weight[expert, :, :]
            scales = w2_amax[expert] / vmax
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            w2[expert, :, :] = q.to(torch.int8)
            w2_scale[expert, :] = scales

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_scale", w13_scale)
        replace_parameter(layer, "w2_scale", w2_scale)

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        w13, w2 = convert_to_int8_moe_kernel_format(
            int8_backend=self.int8_backend,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            layer=layer,
            w13_scale=layer.w13_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_int8_moe_kernel(
            int8_backend=self.int8_backend,
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        return make_int8_moe_quant_config(
            int8_backend=self.int8_backend,
            w1_scale=getattr(layer, "w13_scale", None),
            w2_scale=getattr(layer, "w2_scale", None),
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            per_out_ch_quant=True,
            layer=layer,
        )


def quantize_int8_per_group_sym(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 quantization of the rows of `weight` [N, K].

    Returns the codes [N, K] int8 and the scales [N, K // group_size] in the
    weight dtype ([N, 1] for `group_size` -1, one scale per output channel),
    such that `codes * scale` approximates `weight`.
    """
    n, k = weight.shape
    groups = weight.float().view(n, -1, k if group_size == -1 else group_size)
    # Round against the stored scale, so the codes match what the kernel
    # dequantizes.
    scale = (
        (groups.abs().amax(dim=-1) / _INT8_MAX)
        .clamp(min=torch.finfo(weight.dtype).tiny)
        .to(weight.dtype)
    )
    codes = (groups / scale.float().unsqueeze(-1)).round().clamp(-_INT8_MAX, _INT8_MAX)
    return codes.to(torch.int8).view(n, k), scale


@torch.no_grad()
def quantize_int8_to_mp_layout(
    weight: torch.Tensor, group_size: int, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Quantize `weight` [N, K] into the `MPLinearKernel` `uint8b128` layout.

    Returns the codes offset by 128 and packed along K (int32 [N, K // 4]),
    the scales from `quantize_int8_per_group_sym` and no zero points, the
    layout of compressed-tensors symmetric W8A16 checkpoints. Rows are
    processed in chunks to bound the fp32 working set.
    """
    n, k = weight.shape
    num_groups = 1 if group_size == -1 else k // group_size
    packed = torch.empty(n, k // _PACK_FACTOR, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, num_groups, dtype=weight.dtype, device=weight.device)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        codes, scales[rows] = quantize_int8_per_group_sym(weight[rows], group_size)
        packed[rows] = pack_quantized_values_into_int32(
            codes.to(torch.int32) + scalar_types.uint8b128.bias,
            scalar_types.uint8b128,
            packed_dim=1,
        )
    return packed, scales, None


class Int8PerChannelLMHeadMethod(OnlineLMHeadMethodBase):
    """Weight-only symmetric int8 lm_head, one scale per output channel."""

    scheme = "int8_per_channel_static"
    weight_quant_key = kInt8StaticChannelSym
    activation_quant_key = None
    weight_type = scalar_types.uint8b128
    group_size = -1
    zero_points = False
    k_align = _PACK_FACTOR

    def quantize(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        return quantize_int8_to_mp_layout(weight, self.group_size)


class Int8PerGroup32LMHeadMethod(Int8PerChannelLMHeadMethod):
    """Weight-only symmetric int8 lm_head, one scale per 32 input channels."""

    scheme = "int8_per_group_32"
    weight_quant_key = kInt8Static32Sym
    group_size = 32
    k_align = group_size
