# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


def _make_fake_fused_moe(*, is_act_and_mul: bool = True) -> FusedMoE:
    moe = object.__new__(FusedMoE)
    moe.moe_config = SimpleNamespace(is_act_and_mul=is_act_and_mul)
    return moe


def test_load_w13_with_rank_local_weight_does_not_reshard():
    moe = _make_fake_fused_moe()
    expert_data = torch.zeros(4096, 8)
    loaded_weight = torch.arange(2048 * 8, dtype=torch.float32).reshape(2048, 8)

    FusedMoE._load_w13(
        moe,
        expert_data=expert_data,
        shard_dim=0,
        shard_id="w3",
        loaded_weight=loaded_weight,
        tp_rank=1,
        load_full=False,
    )

    assert torch.equal(expert_data[2048:], loaded_weight)
    assert torch.count_nonzero(expert_data[:2048]) == 0


def test_load_w13_with_global_weight_still_shards_by_tp_rank():
    moe = _make_fake_fused_moe()
    expert_data = torch.zeros(4096, 8)
    loaded_weight = torch.arange(4096 * 8, dtype=torch.float32).reshape(4096, 8)

    FusedMoE._load_w13(
        moe,
        expert_data=expert_data,
        shard_dim=0,
        shard_id="w1",
        loaded_weight=loaded_weight,
        tp_rank=1,
        load_full=False,
    )

    assert torch.equal(expert_data[:2048], loaded_weight[2048:])
    assert torch.count_nonzero(expert_data[2048:]) == 0


def test_load_w2_with_rank_local_weight_does_not_reshard():
    moe = _make_fake_fused_moe()
    expert_data = torch.zeros(8, 2048)
    loaded_weight = torch.arange(8 * 2048, dtype=torch.float32).reshape(8, 2048)

    FusedMoE._load_w2(
        moe,
        expert_data=expert_data,
        shard_dim=1,
        loaded_weight=loaded_weight,
        tp_rank=1,
        load_full=False,
    )

    assert torch.equal(expert_data, loaded_weight)


def test_load_w13_non_act_and_mul_rank_local_does_not_reshard():
    moe = _make_fake_fused_moe(is_act_and_mul=False)
    expert_data = torch.zeros(2048, 8)
    loaded_weight = torch.arange(2048 * 8, dtype=torch.float32).reshape(2048, 8)

    FusedMoE._load_w13(
        moe,
        expert_data=expert_data,
        shard_dim=0,
        shard_id="w1",
        loaded_weight=loaded_weight,
        tp_rank=3,
        load_full=False,
    )

    assert torch.equal(expert_data, loaded_weight)


def test_load_w2_load_full_keeps_global_tensor():
    moe = _make_fake_fused_moe()
    expert_data = torch.zeros(8, 4096)
    loaded_weight = torch.arange(8 * 4096, dtype=torch.float32).reshape(8, 4096)

    FusedMoE._load_w2(
        moe,
        expert_data=expert_data,
        shard_dim=1,
        loaded_weight=loaded_weight,
        tp_rank=1,
        load_full=True,
    )

    assert torch.equal(expert_data, loaded_weight)
