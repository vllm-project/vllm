# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearLayerConfig,
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)


def _fp8_linear_config(
    activation_group_shape: GroupShape,
) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=QuantKey(
            dtype=torch.float8_e4m3fn,
            scale=ScaleDesc(
                dtype=torch.float32,
                static=True,
                group_shape=GroupShape.PER_CHANNEL,
            ),
        ),
        activation_quant_key=QuantKey(
            dtype=torch.float8_e4m3fn,
            scale=ScaleDesc(
                dtype=torch.float32,
                static=False,
                group_shape=activation_group_shape,
            ),
        ),
        weight_shape=(128, 128),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )


def test_marlin_fp8_refuses_block_fp8_activation_scales():
    can_implement, reason = MarlinFP8ScaledMMLinearKernel.can_implement(
        _fp8_linear_config(GroupShape(1, 128))
    )

    assert not can_implement
    assert reason is not None
    assert "block-FP8" in reason


def test_marlin_fp8_keeps_non_block_fp8_layers_available():
    can_implement, reason = MarlinFP8ScaledMMLinearKernel.can_implement(
        _fp8_linear_config(GroupShape.PER_TOKEN)
    )

    assert can_implement
    assert reason is None
