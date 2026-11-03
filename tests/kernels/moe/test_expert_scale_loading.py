# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor expert scale loading writes the same bytes as index assignment."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

EXPERT_ID = 2


@pytest.fixture
def fp8_static_experts(dist_init, default_vllm_config, workspace_init):
    layer = FusedMoEFactory(
        num_experts=4,
        top_k=2,
        hidden_size=256,
        intermediate_size=512,
        params_dtype=torch.bfloat16,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="static"
        ),
        prefix="model.layers.0.mlp.experts",
    )
    return layer.routed_experts


@pytest.mark.parametrize("scale_shape", [(), (1,)])
@pytest.mark.parametrize(
    "param_name,shard_id",
    [
        ("w13_weight_scale", "w1"),
        ("w13_weight_scale", "w3"),
        ("w2_weight_scale", "w2"),
        ("w13_input_scale", "w1"),
        ("w2_input_scale", "w2"),
    ],
)
def test_per_tensor_scale_matches_index_assignment(
    fp8_static_experts, param_name, shard_id, scale_shape
):
    param = getattr(fp8_static_experts, param_name)
    scale = torch.full(scale_shape, 0.25)

    # What the loaders did before: ``param[expert][shard] = scale``.
    expected = param.data.clone()
    if param_name == "w13_weight_scale":
        expected[EXPERT_ID][0 if shard_id == "w1" else 1] = scale
    else:
        expected[EXPERT_ID] = scale

    assert param.weight_loader(
        param, scale, param_name, shard_id, EXPERT_ID, return_success=True
    )

    torch.testing.assert_close(param.data, expected, rtol=0, atol=0)
