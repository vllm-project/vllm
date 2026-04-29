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


class _MockPrefetchLayer(_MockMoELayer):
    def __init__(self) -> None:
        super().__init__()
        self.moe_offload_cache = None

    def move_moe_offload_cache_to_device(self, device: torch.device) -> None:
        self.moe_offload_cache.ensure_targets_on_device(device)
        self.w13_weight = torch.nn.Parameter(
            self.moe_offload_cache.target_for("w13_weight"),
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
        mode="passive",
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


def test_passive_cache_logs_transfer_not_pager(monkeypatch):
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="passive",
    )
    log_messages = []
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.moe_offload.logger.debug",
        lambda message, *args: log_messages.append(message % args),
    )

    cache.ensure_experts_resident({1: 4, 2: 3})

    assert len(log_messages) == 1
    assert "[MoE CPU Offload] Passive transfer" in log_messages[0]
    assert "active_experts=[1, 2]" in log_messages[0]
    assert "step=0" in log_messages[0]
    assert "MoE expert pager" not in log_messages[0]


def test_expert_cache_rejects_more_demand_than_staging_budget():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="passive",
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
        mode="passive",
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
        mode="passive",
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (50, 100))
    counts = {0: 1, 1: 4, 2: 3, 3: 2}

    waves = cache.expert_batches_for_counts(counts)

    assert cache.active_expert_budget == 2
    assert waves == [{1: 4, 2: 3}, {3: 2, 0: 1}]


def test_expert_cache_auto_budget_does_not_allocate_before_wave_sizing(
    monkeypatch,
):
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=None,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cuda"),
        mode="passive",
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (50, 100))

    waves = cache.expert_batches_for_counts({0: 1, 1: 4, 2: 3, 3: 2})

    assert waves == [{1: 4, 2: 3}, {3: 2, 0: 1}]
    assert cache.active_expert_budget == 2
    assert cache.target_for("w13_weight").shape[0] == 0
    assert cache.resident_expert_ids() == set()


def test_expert_cache_wave_tensors_mask_nonresident_experts():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="passive",
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
        mode="passive",
    )

    cache.ensure_experts_resident({1: 4, 2: 3})
    cache.ensure_experts_resident({3: 2, 0: 1})

    assert cache.resident_expert_ids() == {0, 3}


def test_expert_cache_can_keep_prefetched_experts_resident():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=3,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="prefetch",
    )

    cache.ensure_experts_resident(
        {0: 0, 1: 0},
        evict_unrequested=False,
    )
    cache.ensure_experts_resident(
        {2: 5},
        evict_unrequested=False,
    )

    assert cache.resident_expert_ids() == {0, 1, 2}


def test_expert_cache_protects_current_prefetch_wave_from_eviction():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="prefetch",
    )

    cache.ensure_experts_resident(
        {0: 100, 1: 100},
        evict_unrequested=False,
    )
    cache.ensure_experts_resident(
        {2: 1, 3: 2},
        evict_unrequested=False,
    )

    assert cache.resident_expert_ids() == {2, 3}


def test_prefetch_prepare_marks_missing_without_loading():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="prefetch",
    )
    cache.ensure_experts_resident({0: 10}, evict_unrequested=False)

    waves = cache.prepare_prefetch_request({0: 3, 2: 4})

    assert waves == [{0: 3}]
    assert cache.resident_expert_ids() == {0}
    assert cache.working_experts == {0, 2}
    assert cache.missing_experts == {2}


def test_prefetch_pager_loads_missing_without_mutating_missing_list():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="prefetch",
    )

    assert cache.prepare_prefetch_request({2: 4}) == []
    assert cache.pager_step()

    assert cache.resident_expert_ids() == {2}
    assert cache.missing_experts == {2}
    assert cache.active_experts[2].gpu_slot_id == 0
    assert cache._free_slots == [1]
    assert torch.equal(cache.target_for("w13_weight")[0], layer.w13_weight[2])


def test_prefetch_pager_does_not_evict_working_expert():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="prefetch",
    )
    cache.ensure_experts_resident({0: 10, 1: 1}, evict_unrequested=False)

    waves = cache.prepare_prefetch_request({0: 5, 2: 4})
    assert waves == [{0: 5}]
    assert cache.pager_step()

    assert cache.resident_expert_ids() == {0, 2}
    assert cache.active_experts[0].gpu_slot_id == 0
    assert cache.active_experts[2].gpu_slot_id == 1


def test_prefetch_finish_clears_completed_working_entries():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="prefetch",
    )
    cache.prepare_prefetch_request({1: 4, 2: 3})

    cache.finish_prefetch_request({1, 2})

    assert cache.working_experts == set()
    assert cache.missing_experts == {1, 2}


def test_prefetch_prepare_can_wait_for_pager_loaded_wave():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="prefetch",
    )
    cache.start_prefetch_pager()

    try:
        waves = cache.prepare_prefetch_request(
            {1: 4, 2: 3},
            wait_for_resident=True,
        )
    finally:
        cache.stop_prefetch_pager()

    assert waves == [{1: 4, 2: 3}]
    assert cache.resident_expert_ids() == {1, 2}
    assert cache.missing_experts == {1, 2}


def test_prefetch_runtime_replaces_cpu_placeholder_with_execution_target():
    layer = _MockPrefetchLayer()
    layer.moe_offload_cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="prefetch",
    )
    layer.w13_weight = torch.nn.Parameter(
        layer.moe_offload_cache.target_for("w13_weight"),
        requires_grad=False,
    )

    assert layer.w13_weight.shape[0] == 0

    layer.move_moe_offload_cache_to_device(torch.device("cpu"))

    assert layer.w13_weight.shape == (2, 3, 2)
    assert layer.w13_weight.data_ptr() == layer.moe_offload_cache.target_for(
        "w13_weight"
    ).data_ptr()


def test_expert_cache_retires_loaded_experts():
    layer = _MockMoELayer()
    cache = ExpertCache.from_cpu_sources(
        layer_id=3,
        active_expert_budget=2,
        sources={"w13_weight": layer.w13_weight.detach()},
        device=torch.device("cpu"),
        mode="passive",
    )

    cache.ensure_experts_resident({1: 4, 2: 3})
    cache.retire_experts({1, 2})

    assert cache.resident_expert_ids() == set()
    assert cache._free_slots == [0, 1]
