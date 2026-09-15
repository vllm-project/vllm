# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
    _make_fused_conv1d_weight_loader,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.utils import set_weight_attrs


def test_fused_conv1d_loader_uses_stacked_mapper_shard_id():
    # Given
    dims = [4, 4, 6]
    weights = [
        ("q_conv1d", torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)),
        ("k_conv1d", torch.arange(16, 32, dtype=torch.float32).reshape(4, 1, 4)),
        ("v_conv1d", torch.arange(32, 56, dtype=torch.float32).reshape(6, 1, 4)),
    ]
    expected = torch.cat([weights[0][1][2:], weights[1][1][2:], weights[2][1][3:]])
    module = torch.nn.Module()
    param = torch.nn.Parameter(torch.full_like(expected, -1.0))
    set_weight_attrs(
        param,
        {
            "weight_loader": _make_fused_conv1d_weight_loader(
                dims=dims,
                tp_size=2,
                tp_rank=1,
            )
        },
    )
    module.register_parameter("conv1d", param)
    mapper = WeightsMapper(
        orig_to_new_stacked={
            "q_conv1d": ("conv1d", 0),
            "k_conv1d": ("conv1d", 1),
            "v_conv1d": ("conv1d", 2),
        }
    )

    # When
    loaded_params = AutoWeightsLoader(module).load_weights(weights, mapper=mapper)

    # Then
    assert loaded_params == {"conv1d"}
    torch.testing.assert_close(param, expected)
