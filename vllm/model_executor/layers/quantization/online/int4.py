# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online int4 weight-only quantization.

`Int4PerGroup32AsymLMHeadMethod` quantizes an fp16/bf16 lm_head at load time
(a `quantization_config.targets` pattern naming the lm_head, with
`{"weight": "int4_per_group_32_asym"}`). Decode reads the
whole lm_head every step, so on bandwidth-bound GPUs a full-precision head is
one of the largest per-token costs even when the checkpoint quantizes every
other linear layer.

`Int4PerGroup32OnlineMoEMethod` quantizes routed experts to symmetric int4
with one scale per 32 input channels (`{"moe": {"weight": "int4_per_group_32"}}`),
from fp16/bf16 weights or by re-quantizing a checkpoint's quantized experts.
"""

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    WNA16WeightFormat,
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    make_wna16_moe_quant_config,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.online.lm_head import (
    OnlineLMHeadMethodBase,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt4Static32,
    kInt4Static32Asym,
    pack_quantized_values_into_int32,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

_UINT4_MAX = 15
_INT4_SYM_MAX = 7
# Zero points pack 8 output rows per int32 word.
_PACK_FACTOR = 8
# Experts are quantized in chunks of about this many values, which bounds the
# staging when a full-precision layer is quantized at once.
_MOE_QUANT_CHUNK_VALUES = 1 << 27


def quantize_int4_per_group_asym(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric uint4 quantization of the rows of `weight` [N, K].

    Returns the codes [N, K] uint8, the scales [N, K // group_size] in the
    weight dtype and the zero points [N, K // group_size] uint8, such that
    `(codes - zero_point) * scale` approximates `weight`. Each group's
    range is widened to include 0, so the zero point always fits in uint4.
    """
    n, k = weight.shape
    groups = weight.float().view(n, k // group_size, group_size)
    w_min = groups.amin(dim=-1).clamp(max=0)
    w_max = groups.amax(dim=-1).clamp(min=0)
    # Round against the stored scale, so the codes match what the kernel
    # dequantizes.
    scale = (
        ((w_max - w_min) / _UINT4_MAX)
        .clamp(min=torch.finfo(weight.dtype).tiny)
        .to(weight.dtype)
    )
    scale_f = scale.float()
    zero_point = (-w_min / scale_f).round().clamp(0, _UINT4_MAX)
    codes = ((groups / scale_f.unsqueeze(-1)).round() + zero_point.unsqueeze(-1)).clamp(
        0, _UINT4_MAX
    )
    return codes.to(torch.uint8).view(n, k), scale, zero_point.to(torch.uint8)


@torch.no_grad()
def quantize_int4_to_mp_layout(
    weight: torch.Tensor, group_size: int, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize `weight` [N, K] into the `MPLinearKernel` uint4 layout.

    Returns the codes packed along K (int32 [N, K // 8]), the scales
    [N, K // group_size] and the zero points packed along N
    (int32 [N // 8, K // group_size]), the layout of compressed-tensors
    asymmetric W4A16 checkpoints. Rows are processed in chunks to bound the
    fp32 working set.
    """
    n, k = weight.shape
    packed = torch.empty(n, k // _PACK_FACTOR, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // group_size, dtype=weight.dtype, device=weight.device)
    zero_points = torch.empty(
        n, k // group_size, dtype=torch.uint8, device=weight.device
    )
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        codes, scales[rows], zero_points[rows] = quantize_int4_per_group_asym(
            weight[rows], group_size
        )
        packed[rows] = pack_quantized_values_into_int32(
            codes.to(torch.int32), scalar_types.uint4, packed_dim=1
        )
    packed_zero_points = pack_quantized_values_into_int32(
        zero_points.to(torch.int32), scalar_types.uint4, packed_dim=0
    )
    return packed, scales, packed_zero_points


def quantize_int4_per_group_sym(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int4 quantization of `weight` [..., N, K] along K.

    Returns the codes as uint4b8 packed along K (int32 [..., N, K // 8]) and
    the scales [..., N, K // group_size] in the weight dtype, such that
    `(code - 8) * scale` approximates `weight`. Codes stay in [-7, 7]. The math
    runs in the weight dtype, like AITER's `per_1x32_i4_quant`, whose codes and
    scales it reproduces for bf16 weights.
    """
    *batch, n, k = weight.shape
    groups = weight.view(*batch, n, k // group_size, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / _INT4_SYM_MAX
    codes = (groups / scale).round().clamp(-_INT4_SYM_MAX, _INT4_SYM_MAX)
    codes = codes.to(torch.int32).view(*batch, n, k) + scalar_types.uint4b8.bias
    packed = pack_quantized_values_into_int32(
        codes, scalar_types.uint4b8, packed_dim=codes.dim() - 1
    )
    return packed, scale.squeeze(-1)


def _quantize_int4_moe_weight(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`quantize_int4_per_group_sym` of stacked `[E, N, K]` expert weights."""
    num_experts, n, k = weight.shape
    packed = torch.empty(
        (num_experts, n, k // _PACK_FACTOR), dtype=torch.int32, device=weight.device
    )
    scale = torch.empty(
        (num_experts, n, k // group_size), dtype=weight.dtype, device=weight.device
    )
    chunk = max(1, _MOE_QUANT_CHUNK_VALUES // (n * k))
    for start in range(0, num_experts, chunk):
        experts = slice(start, start + chunk)
        packed[experts], scale[experts] = quantize_int4_per_group_sym(
            weight[experts], group_size
        )
    return packed, scale


class Int4PerGroup32AsymLMHeadMethod(OnlineLMHeadMethodBase):
    """Weight-only asymmetric uint4 lm_head, one scale and zero point per 32
    input channels."""

    scheme = "int4_per_group_32_asym"
    weight_quant_key = kInt4Static32Asym
    activation_quant_key = None
    weight_type = scalar_types.uint4
    group_size = 32
    zero_points = True
    n_align = _PACK_FACTOR
    k_align = group_size

    def quantize(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return quantize_int4_to_mp_layout(weight, self.group_size)


class Int4PerGroup32OnlineMoEMethod(OnlineMoEMethodBase):
    """Online weight-only symmetric int4 MoE, one scale per 32 input channels,
    on the WNA16 MoE oracle."""

    weight_format = WNA16WeightFormat(num_bits=4, group_size=32)
    activation_quant_key = None
    experts_cls: "type[mk.FusedMoEExperts]"

    def __init__(self, *, moe: FusedMoEConfig):
        # The CUDA WNA16 backends have not been validated with this method yet.
        if not current_platform.is_rocm():
            raise NotImplementedError(
                "Online int4_per_group_32 MoE quantization is currently only "
                "supported on ROCm."
            )
        super().__init__(moe)
        self.wna16_backend, self.experts_cls = select_wna16_moe_backend(
            config=moe,
            weight_key=kInt4Static32,
            quant_config=self.weight_format,
            may_have_zp=False,
            may_have_bias=moe.has_bias,
            allow_tile_padding=True,
        )

    @property
    def supports_eplb(self) -> bool:
        return self.wna16_backend == WNA16MoEBackend.TRITON

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        group_size = self.weight_format.group_size
        if hidden_size % group_size or intermediate_size_per_partition % group_size:
            raise ValueError(
                f"Online int4 MoE requires the hidden size ({hidden_size}) and "
                "intermediate size per partition "
                f"({intermediate_size_per_partition}) to be divisible by "
                f"{group_size}."
            )
        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.w13_input_scale = None
        layer.w2_input_scale = None
        group_size = self.weight_format.group_size

        def quantize(w13_weight: torch.Tensor, w2_weight: torch.Tensor):
            self._zero_padding(layer, w13_weight, w2_weight)
            return (
                *_quantize_int4_moe_weight(w13_weight, group_size),
                *_quantize_int4_moe_weight(w2_weight, group_size),
            )

        w13, w13_scale, w2, w2_scale = self.quantize_weights(layer, quantize)
        # Humming converts the layer's parameters in place.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        del w13, w2, w13_scale, w2_scale

        assert isinstance(layer, RoutedExperts)
        self._setup_kernel(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        converted = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_backend,
            layer=layer,
            quant_config=self.weight_format,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )
        if converted is not None:
            w13, w2, w13_scale, w2_scale, _, _, w13_bias, w2_bias = converted
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
            if w13_bias is not None and w2_bias is not None:
                # Detached, so a bias passed through unchanged does not keep the
                # source's marker and get released with its parameters.
                replace_parameter(layer, "w13_bias", w13_bias.detach())
                replace_parameter(layer, "w2_bias", w2_bias.detach())

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.wna16_backend,
            routing_tables=layer._expert_routing_tables(),
        )

    def get_fused_moe_quant_config(self, layer: Module) -> FusedMoEQuantConfig:
        swiglu = {
            "gemm1_alpha": getattr(layer, "swiglu_alpha", None),
            "gemm1_beta": getattr(layer, "swiglu_beta", None),
            "gemm1_clamp_limit": getattr(layer, "swiglu_limit", None),
        }
        if self.wna16_backend == WNA16MoEBackend.HUMMING:
            from vllm.model_executor.layers.quantization.utils.humming import (
                get_humming_moe_quant_config,
            )

            return get_humming_moe_quant_config(layer, **swiglu)
        return make_wna16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_size=self.weight_format.group_size,
            num_bits=self.weight_format.num_bits,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            **swiglu,
        )
