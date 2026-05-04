# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

def test_mxfp4_e8m0_scale_loading_preserves_raw_bytes():
    from types import SimpleNamespace

    import pytest
    import torch

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("torch does not expose float8_e8m0fnu")

    layer = object.__new__(FusedMoE)
    layer.moe_config = SimpleNamespace(is_act_and_mul=True)

    expert_data = torch.zeros((4, 2), dtype=torch.uint8)
    loaded_scale = torch.tensor(
        [[0.0078125, 0.015625], [0.5, 1.0]],
        dtype=e8m0_dtype,
    )

    layer._load_w13(
        expert_data=expert_data,
        shard_dim=0,
        shard_id="w1",
        loaded_weight=loaded_scale,
        tp_rank=0,
    )

    torch.testing.assert_close(
        expert_data[:2],
        loaded_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )
