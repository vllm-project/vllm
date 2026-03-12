# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.grok1 import Grok1MoE


def _try_load_pre_sharded_expert_weight(
    moe: Grok1MoE,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    expert_id: int,
) -> bool:
    return Grok1MoE.try_load_pre_sharded_expert_weight(
        moe, param, loaded_weight, shard_id, expert_id
    )


def _make_fake_moe(*, is_act_and_mul: bool = True) -> Grok1MoE:
    moe = object.__new__(Grok1MoE)
    moe.experts = SimpleNamespace(
        moe_config=SimpleNamespace(is_act_and_mul=is_act_and_mul),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
    )
    return moe


def test_pre_sharded_w3_loads_directly_into_local_slice():
    moe = _make_fake_moe()
    param = torch.nn.Parameter(torch.zeros(1, 4096, 8))
    loaded_weight = torch.arange(2048 * 8, dtype=torch.float32).reshape(2048, 8)

    loaded = _try_load_pre_sharded_expert_weight(
        moe, param, loaded_weight, "w3", expert_id=0
    )

    assert loaded
    assert torch.equal(param.data[0, 2048:], loaded_weight)
    assert torch.count_nonzero(param.data[0, :2048]) == 0


def test_global_w3_weight_falls_back_to_generic_loader():
    moe = _make_fake_moe()
    param = torch.nn.Parameter(torch.zeros(1, 4096, 8))
    loaded_weight = torch.arange(4096 * 8, dtype=torch.float32).reshape(4096, 8)

    loaded = _try_load_pre_sharded_expert_weight(
        moe, param, loaded_weight, "w3", expert_id=0
    )

    assert not loaded


def test_pre_sharded_w2_loads_directly():
    moe = _make_fake_moe()
    param = torch.nn.Parameter(torch.zeros(1, 8, 2048))
    loaded_weight = torch.arange(8 * 2048, dtype=torch.float32).reshape(8, 2048)

    loaded = _try_load_pre_sharded_expert_weight(
        moe, param, loaded_weight, "w2", expert_id=0
    )

    assert loaded
    assert torch.equal(param.data[0], loaded_weight)
