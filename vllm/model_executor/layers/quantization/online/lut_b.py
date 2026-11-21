# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online LUT-B quantization of routed MoE experts (MXFP8 activations)."""

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lut_b_moe import (
    LutBSm107Experts,
    make_lut_b_moe_kernel,
    select_lut_b_moe_experts,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    dequantize_lut_b,
    lut_b_to_sm107_tiles,
    quantize_lut_b,
)
from vllm.model_executor.utils import replace_parameter


class LutBOnlineMoEMethod(OnlineMoEMethodBase):
    """Quantize BF16/FP16 routed experts to LUT-B while loading.

    Weights use 3-bit indices into one E4M3 codebook (with a UE8M0 scale) per
    8x64 tile and activations are MXFP8. Rubin runs the native LUT-B tensor
    core kernel; other GPUs emulate the same numerics.
    """

    def __init__(self, *, moe: FusedMoEConfig):
        super().__init__(moe)
        if moe.has_bias:
            raise ValueError("LUT-B online MoE quantization does not support bias.")

        self.experts_cls = select_lut_b_moe_experts(moe)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # Both experts quantize activations themselves, and the native weights
        # carry their codebooks and scales inline.
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13, w2 = self.get_weights_for_quantization(layer)
        self._zero_padding(layer, w13, w2)
        for name, weight in (("w13_weight", w13), ("w2_weight", w2)):
            packed, codebooks, scales = quantize_lut_b(weight)
            if self.experts_cls is LutBSm107Experts:
                weight = lut_b_to_sm107_tiles(packed, codebooks, scales)
            else:
                weight = dequantize_lut_b(packed, codebooks, scales, weight.dtype)
            replace_parameter(layer, name, weight)
        self.release_requantization_source_weights(layer)

        quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant_config
        self.moe_kernel = make_lut_b_moe_kernel(
            self.experts_cls,
            self.moe,
            quant_config,
            layer._expert_routing_tables(),
        )
        layer._already_called_process_weights_after_loading = True
