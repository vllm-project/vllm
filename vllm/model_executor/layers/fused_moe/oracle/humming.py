# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.utils import replace_parameter


def stage_moe_weights_for_humming(
    layer: RoutedExperts,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None = None,
    w2_input_scale: torch.Tensor | None = None,
) -> str:
    """Install canonical weights and scales for Humming's layer-based converter."""
    scale_name = (
        "weight_scale_inv" if hasattr(layer, "w13_weight_scale_inv") else "weight_scale"
    )
    replace_parameter(layer, "w13_weight", w13)
    replace_parameter(layer, "w2_weight", w2)
    replace_parameter(layer, f"w13_{scale_name}", w13_scale)
    replace_parameter(layer, f"w2_{scale_name}", w2_scale)
    if w13_input_scale is not None:
        replace_parameter(layer, "w13_input_scale", w13_input_scale)
    if w2_input_scale is not None:
        replace_parameter(layer, "w2_input_scale", w2_input_scale)
    return scale_name
