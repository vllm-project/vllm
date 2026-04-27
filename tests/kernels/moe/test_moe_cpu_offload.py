# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.moe_offload import (
    ExpertCache,
    local_expert_token_counts,
)


class _MockMoELayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_num_experts = 4
        self.w13_weight = torch.nn.Parameter(
            torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.arange(100, 100 + 4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
            requires_grad=False,
        )
        self.router_weight = torch.nn.Parameter(
            torch.ones(2, 2),
            requires_grad=False,
        )


def test_local_expert_token_counts_without_expert_map():
    topk_ids = torch.tensor([[2, 1], [2, -1], [3, 2], [9, 1]])

    counts = local_expert_token_counts(
        topk_ids,
        local_num_experts=4,
        expert_map=None,
    )

    assert counts == {1: 2, 2: 3, 3: 1}


def test_local_expert_token_counts_with_expert_map():
    topk_ids = torch.tensor([[0, 1], [2, 3], [4, 1]])
    expert_map = torch.tensor([3, -1, 0, 2])

    counts = local_expert_token_counts(
        topk_ids,
        local_num_experts=4,
        expert_map=expert_map,
    )

    assert counts == {0: 1, 2: 1, 3: 1}


def test_expert_cache_restores_expert_slices_from_cpu_source():
    layer = _MockMoELayer()
    cache = ExpertCache.from_layer(
        layer,
        active_expert_budget=2,
        layer_id=7,
    )
    expected_w13_expert_2 = layer.w13_weight[2].detach().clone()
    expected_w2_expert_2 = layer.w2_weight[2].detach().clone()

    layer.w13_weight.data[2].zero_()
    layer.w2_weight.data[2].zero_()
    cache.ensure_experts_resident({2: 5})

    assert torch.equal(layer.w13_weight[2], expected_w13_expert_2)
    assert torch.equal(layer.w2_weight[2], expected_w2_expert_2)
    assert cache.resident_expert_ids() == {2}
    assert cache.active_experts[2].layer_id == 7
    assert cache.active_experts[2].recent_token_count == 5


def test_expert_cache_enforces_budget_with_coldest_oldest_eviction():
    layer = _MockMoELayer()
    cache = ExpertCache.from_layer(
        layer,
        active_expert_budget=2,
        layer_id=0,
    )

    cache.ensure_experts_resident({0: 5})
    cache.ensure_experts_resident({1: 1})
    cache.ensure_experts_resident({2: 3})

    assert cache.resident_expert_ids() == {0, 2}


def test_expert_cache_stages_cpu_sources_and_remaps_topk_ids():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={
            "w13_weight": layer.w13_weight.detach(),
            "w2_weight": layer.w2_weight.detach(),
        },
        device=torch.device("cpu"),
    )

    topk_ids = torch.tensor([[2, 1], [2, 1]])
    remapped = cache.ensure_experts_resident_and_remap(
        topk_ids,
        local_num_experts=4,
        expert_map=None,
    )

    assert cache.target_for("w13_weight").shape == (2, 3, 2)
    assert torch.equal(cache.target_for("w13_weight")[0], layer.w13_weight[1])
    assert torch.equal(cache.target_for("w13_weight")[1], layer.w13_weight[2])
    assert torch.equal(remapped, torch.tensor([[1, 0], [1, 0]]))
    assert cache.active_experts[1].gpu_slot_id == 0
    assert cache.active_experts[2].gpu_slot_id == 1


def test_expert_cache_rejects_more_demand_than_staging_budget():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
    )

    topk_ids = torch.tensor([[0, 1], [2, 1]])

    try:
        cache.ensure_experts_resident_and_remap(
            topk_ids,
            local_num_experts=4,
            expert_map=None,
        )
    except RuntimeError as exc:
        assert "active_expert_staging_slots=2" in str(exc)
    else:
        raise AssertionError("Expected active expert budget failure")


def test_expert_cache_builds_budget_sized_waves():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
    )
    counts = {0: 1, 1: 4, 2: 3, 3: 2}

    waves = cache.expert_batches_for_counts(counts)

    assert waves == [{1: 4, 2: 3}, {3: 2, 0: 1}]


def test_expert_cache_auto_budget_uses_smaller_gpu_memory_wave(monkeypatch):
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=None,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cuda"),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (50, 100))
    counts = {0: 1, 1: 4, 2: 3, 3: 2}

    waves = cache.expert_batches_for_counts(counts)

    assert cache.active_expert_budget == 2
    assert waves == [{1: 4, 2: 3}, {3: 2, 0: 1}]


def test_expert_cache_wave_tensors_mask_nonresident_experts():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
    )
    topk_ids = torch.tensor([[1, 2], [3, 0]])
    topk_weights = torch.tensor([[0.4, 0.6], [0.25, 0.75]])

    cache.ensure_experts_resident({1: 4, 2: 3})
    wave_ids, wave_weights = cache.make_wave_tensors(
        topk_ids,
        topk_weights,
        local_expert_ids={1, 2},
        expert_map=None,
    )

    assert torch.equal(wave_ids, torch.tensor([[0, 1], [0, 0]]))
    assert torch.equal(wave_weights, torch.tensor([[0.4, 0.6], [0.0, 0.0]]))


def test_expert_cache_keeps_requested_wave_resident_when_reusing_slots():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
    )

    cache.ensure_experts_resident({1: 4, 2: 3})
    cache.ensure_experts_resident({3: 2, 0: 1})

    assert cache.resident_expert_ids() == {0, 3}


def test_expert_cache_retires_loaded_experts():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
    )

    cache.ensure_experts_resident({1: 4, 2: 3})
    cache.retire_experts({1, 2})

    assert cache.resident_expert_ids() == set()
    assert cache._free_slots == [0, 1]
