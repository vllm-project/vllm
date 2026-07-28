# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hierarchical (Colibri-style) expert staging."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vllm.config.offload import HierarchicalOffloadConfig, OffloadConfig
from vllm.model_executor.offloader.base import create_offloader
from vllm.model_executor.offloader.hierarchical.device_slots import ExpertSlotPool
from vllm.model_executor.offloader.hierarchical.format import (
    convert_layer_from_device_params,
    load_manifest,
    pack_expert_row_torch,
    unpack_expert_row,
)
from vllm.model_executor.offloader.hierarchical.planner import (
    build_tier_plan,
    compute_slots_per_layer,
    resolve_ram_budget_bytes,
)
from vllm.model_executor.offloader.hierarchical.ram_cache import PinnedExpertRamCache
from vllm.model_executor.offloader.hierarchical.usage import ExpertUsageStore
from vllm.model_executor.offloader.hierarchical_offloader import HierarchicalOffloader
from vllm.platforms import current_platform


class _FakeExperts(nn.Module):
    def __init__(self, num_experts: int = 8, hidden: int = 16, inter: int = 32):
        super().__init__()
        self.top_k = 2
        self.w13_weight = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden))
        self.w2_weight = nn.Parameter(torch.randn(num_experts, hidden, inter))


def test_create_offloader_hierarchical():
    cfg = OffloadConfig(
        offload_backend="hierarchical",
        hierarchical=HierarchicalOffloadConfig(tier_num_slots=4),
    )
    off = create_offloader(cfg)
    assert isinstance(off, HierarchicalOffloader)


def test_create_offloader_auto_hierarchical():
    cfg = OffloadConfig(
        offload_backend="auto",
        hierarchical=HierarchicalOffloadConfig(tier_device_expert_gb=1.0),
    )
    off = create_offloader(cfg)
    assert isinstance(off, HierarchicalOffloader)


def test_planner_slots_and_ram():
    cfg = HierarchicalOffloadConfig(tier_num_slots=16, tier_ram_gb=2.0)
    assert compute_slots_per_layer(
        cfg, num_moe_layers=4, num_local_experts=64, expert_row_bytes=1024
    ) == 16
    assert resolve_ram_budget_bytes(cfg) == int(2 * 1024**3)
    plan = build_tier_plan(
        cfg,
        num_moe_layers=4,
        num_local_experts=64,
        expert_row_bytes=1024 * 1024,
        top_k=8,
    )
    assert plan.slots_per_layer == 16
    assert "Hierarchical expert tier plan" in plan.summary()


def test_usage_store_roundtrip(tmp_path: Path):
    path = tmp_path / ".vllm_expert_usage"
    store = ExpertUsageStore(str(path))
    store.record(0, [1, 2, 2, 3])
    store.record(0, [2])
    store.flush()
    store2 = ExpertUsageStore(str(path))
    hot = store2.hottest(0, 2, 8)
    assert hot[0] == 2


def test_ram_cache_put_get_evict():
    row = torch.arange(64, dtype=torch.uint8)
    cache = PinnedExpertRamCache(capacity_bytes=64 * 2, row_nbytes=64)
    assert cache.enabled
    cache.put(0, 0, row, pinned=True)
    cache.put(0, 1, row + 1)
    got = cache.get(0, 0)
    assert got is not None
    assert torch.equal(got, row)
    # Force eviction of non-pinned
    cache.put(0, 2, row + 2)
    # expert 1 may be evicted
    assert cache.get(0, 0) is not None  # pinned survives


def test_expert_store_format(tmp_path: Path):
    w13 = torch.randn(4, 8, 4)
    w2 = torch.randn(4, 4, 8)
    meta = convert_layer_from_device_params(
        tmp_path, layer_id=0, weight_tensors=[w13, w2], model_id="test"
    )
    assert meta.num_experts == 4
    manifest = load_manifest(tmp_path)
    assert manifest is not None
    assert len(manifest.layers) == 1
    packed = pack_expert_row_torch([w13[1], w2[1]])
    specs = meta.tensor_specs
    unpacked = unpack_expert_row(packed, specs)
    assert torch.allclose(unpacked[0], w13[1])
    assert torch.allclose(unpacked[1], w2[1])


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU for device slot DMA",
)
def test_slot_pool_ensure_remap():
    device = torch.device(f"{current_platform.device_type}:0")
    E, H, I = 8, 16, 32
    host_w13 = torch.randn(E, 2 * I, H)
    host_w2 = torch.randn(E, H, I)
    stream = current_platform.Stream()
    pool = ExpertSlotPool(
        layer_id=0,
        weight_templates=[host_w13, host_w2],
        num_slots=4,
        copy_stream=stream,
        device=device,
    )
    ids = [0, 3, 5]
    host_rows = {e: [host_w13[e], host_w2[e]] for e in ids}
    remap, events = pool.ensure_from_host_rows(ids, host_rows)
    compute = current_platform.current_stream()
    for ev in events:
        compute.wait_event(ev)
    assert set(remap.keys()) == set(ids)
    for e, s in remap.items():
        assert torch.allclose(pool.slot_weights[0][s].cpu(), host_w13[e], atol=1e-5)


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU for device slot DMA",
)
def test_slot_pool_protects_same_batch_residents():
    """Same-batch ensure must not evict experts already selected this call."""
    device = torch.device(f"{current_platform.device_type}:0")
    E, H, I = 8, 8, 8
    host_w13 = torch.randn(E, 2 * I, H)
    host_w2 = torch.randn(E, H, I)
    stream = current_platform.Stream()
    pool = ExpertSlotPool(
        layer_id=0,
        weight_templates=[host_w13, host_w2],
        num_slots=2,
        copy_stream=stream,
        device=device,
    )
    # Fill both slots.
    first = [0, 1]
    host_rows = {e: [host_w13[e], host_w2[e]] for e in range(E)}
    remap, events = pool.ensure_from_host_rows(first, host_rows)
    compute = current_platform.current_stream()
    for ev in events:
        compute.wait_event(ev)
    assert pool.contains(0) and pool.contains(1)

    # Request resident 0 plus a new expert: must keep 0, replace 1.
    remap2, events2 = pool.ensure_from_host_rows([0, 2], host_rows)
    for ev in events2:
        compute.wait_event(ev)
    assert set(remap2.keys()) == {0, 2}
    assert pool.contains(0) and pool.contains(2)
    assert not pool.contains(1)
    assert torch.allclose(
        pool.slot_weights[0][remap2[0]].cpu(), host_w13[0], atol=1e-5
    )
    assert torch.allclose(
        pool.slot_weights[0][remap2[2]].cpu(), host_w13[2], atol=1e-5
    )

    # Oversubscribe beyond slots → clear error (not silent corruption).
    with pytest.raises(RuntimeError, match="cannot allocate a slot"):
        pool.ensure_from_host_rows([0, 2, 3], host_rows)


def test_hierarchical_offloader_registers_modules():
    cfg = HierarchicalOffloadConfig(tier_num_slots=2, tier_ram_gb=0.01)
    off = HierarchicalOffloader(cfg)

    def gen():
        for _ in range(2):
            block = nn.Sequential(_FakeExperts())
            # Nest experts so finder sees w13_weight
            yield block

    # Build modules that contain FakeExperts as children
    modules = []
    for i in range(2):
        m = nn.Module()
        m.add_module("mlp", nn.Module())
        m.mlp.add_module("experts", _FakeExperts())  # type: ignore[attr-defined]
        modules.append(m)

    def modules_gen():
        yield from modules

    wrapped = off.wrap_modules(modules_gen())
    assert len(wrapped) == 2
    assert len(off.manager._pending_modules) >= 2
    off.shutdown()
