# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import torch

import vllm.v1.kv_offload.sparse.hisparse_layer as hisparse_layer_module
import vllm.v1.kv_offload.sparse.hisparse_store as hisparse_store_module
import vllm.v1.worker.utils as worker_utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_offload.sparse.base import (
    SparseKVOffloadCommand,
    SparseKVPageTransfer,
)
from vllm.v1.kv_offload.sparse.hisparse_store import (
    HiSparseOffloadStore,
    _expand_source_block_ids,
)
from vllm.v1.metrics.stats import HiSparseStats
from vllm.v1.worker.utils import bind_kv_cache, copy_kv_cache_blocks_inplace


def test_copy_cpu_kv_cache_logical_blocks_ignores_storage_padding(monkeypatch):
    waited_for_host_writes = False

    def wait_for_host_writes():
        nonlocal waited_for_host_writes
        waited_for_host_writes = True

    monkeypatch.setattr(
        worker_utils, "wait_for_hisparse_host_writes", wait_for_host_writes
    )
    backing = torch.full((10, 2, 3), -1, dtype=torch.float32)
    cache = backing[1:9]
    cache[2:4] = 7
    cache[6:8] = 11

    copy_kv_cache_blocks_inplace(
        [cache],
        num_blocks=4,
        kv_cache_block_copies=[
            KVCacheBlockCopy(1, 0),
            KVCacheBlockCopy(3, 2),
        ],
    )

    torch.testing.assert_close(cache[0:2], torch.full_like(cache[0:2], 7))
    torch.testing.assert_close(cache[4:6], torch.full_like(cache[4:6], 11))
    assert waited_for_host_writes
    assert (backing[0] == -1).all()
    assert (backing[9] == -1).all()


def test_expand_hisparse_source_blocks_into_kernel_pages():
    expanded = _expand_source_block_ids([3, 7], blocks_per_kv_block=2, count=3)

    assert expanded.tolist() == [6, 7, 14]


def test_hisparse_layers_join_index_groups_during_construction(monkeypatch):
    """Followers must release duplicate LRU state before memory profiling."""
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        kv_transfer_config=None,
    )
    resolved = hisparse_layer_module.ResolvedHiSparseConfig(
        top_k=4,
        device_buffer_size=8,
        host_pool_gib=1.0,
    )
    monkeypatch.setattr(hisparse_layer_module, "_has_hisparse_ops", lambda: True)
    monkeypatch.setattr(
        hisparse_layer_module.ResolvedHiSparseConfig,
        "from_vllm_config",
        classmethod(lambda cls, vllm_config, model_top_k: resolved),
    )
    monkeypatch.setattr(
        hisparse_layer_module,
        "_get_group_plan",
        lambda device, max_rows, top_k: object(),
    )
    monkeypatch.setattr(
        hisparse_layer_module, "_get_copy_stream", lambda device: object()
    )
    monkeypatch.setattr(hisparse_layer_module, "_CURRENT_INDEX_GROUP", None)
    group_scope = object()

    def make_layer(is_leader: bool):
        layer = hisparse_layer_module.create_hisparse_layer(
            config,
            model_top_k=4,
            index_group_scope=group_scope,
            is_index_group_leader=is_leader,
            row_width=8,
            kv_dtype=torch.float32,
            device="cpu",
        )
        assert layer is not None
        return layer

    first_leader = make_layer(True)
    first_follower = make_layer(False)
    second_leader = make_layer(True)
    second_follower = make_layer(False)

    assert first_follower.offload.leader is first_leader.offload
    assert second_follower.offload.leader is second_leader.offload
    assert first_follower.offload.device_global_indices is None
    assert first_follower.offload.lru_slots is None
    assert second_follower.offload.device_global_indices is None
    assert second_follower.offload.lru_slots is None


def test_hisparse_offload_store_invalidates_only_index_group_leaders(monkeypatch):
    """Recycled blocks must not enter followers whose LRU state was released."""
    store = object.__new__(HiSparseOffloadStore)
    store.kernel_block_size = 2
    calls: list[tuple[str, torch.Tensor]] = []
    leader = SimpleNamespace(
        offload=SimpleNamespace(
            device=torch.device("cpu"),
            leader=None,
            invalidate_slots=lambda slots: calls.append(("leader", slots.clone())),
        )
    )
    follower = SimpleNamespace(
        offload=SimpleNamespace(
            device=torch.device("cpu"),
            leader=leader.offload,
            invalidate_slots=lambda slots: calls.append(("follower", slots.clone())),
        )
    )
    store.layers = [leader, follower]
    store.lru_layers = [leader]
    store._block_staging = torch.empty(1, dtype=torch.long)
    event = SimpleNamespace(synchronize=lambda: None, record=lambda stream: None)
    store._block_staging_event = event
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda device: object())

    store.invalidate_blocks([3])

    assert len(calls) == 1
    assert calls[0][0] == "leader"
    torch.testing.assert_close(calls[0][1], torch.tensor([6, 7]))


def test_hisparse_offload_store_prepare_step_invalidates_and_restores(monkeypatch):
    store = object.__new__(HiSparseOffloadStore)
    store.kernel_block_size = 64
    store._post_forward_transfers = []
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-0": 1, "request-1": 1},
        scheduled_new_reqs=[SimpleNamespace(block_ids=([2, 3],))],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[([4],), None]),
    )
    command = SparseKVOffloadCommand(
        block_table_updates={}, page_transfers=[], fully_resident=True
    )
    layer = SimpleNamespace(fully_resident=False)
    store.layers = [layer]
    calls: list[tuple[Any, ...]] = []
    store.invalidate_blocks = lambda block_ids: calls.append(
        ("invalidate", block_ids, store.kernel_block_size)
    )
    store.restore_prefix = lambda output: calls.append(("restore", output))
    store._enqueue_transfers = lambda transfers: calls.append(("transfer", transfers))

    store.prepare_step(command, scheduler_output)

    assert calls == [
        ("invalidate", [2, 3, 4], 64),
        ("restore", scheduler_output),
    ]
    assert layer.fully_resident
    assert store._post_forward_transfers == []


def test_hisparse_offload_store_enqueues_fused_page_spill(monkeypatch):
    store = object.__new__(HiSparseOffloadStore)
    store.kernel_block_size = 4
    store.blocks_per_kv_block = 2
    store.spill_row_capacity = 8
    store.spill_src_cpu = torch.empty((1, 2, 8), dtype=torch.int64)
    store.spill_dst_cpu = torch.empty((1, 8), dtype=torch.int64)
    store.spill_src_gpu = torch.empty((2, 8), dtype=torch.int64)
    store.spill_dst_gpu = torch.empty(8, dtype=torch.int64)
    store._spill_staging_index = 0
    staging_recorded_streams: list[object] = []
    store._spill_staging_events = [
        SimpleNamespace(query=lambda: True, record=staging_recorded_streams.append)
    ]
    store.spill_src_indices_ptrs = object()
    store.layers = [
        SimpleNamespace(offload=SimpleNamespace(resident_source_index=0)),
        SimpleNamespace(offload=SimpleNamespace(resident_source_index=1)),
    ]
    store._completed_transfer_ids = []
    store.hot_backing = SimpleNamespace(device="cuda:0")
    store.backup_layer_offsets = object()
    store.backup_host_anchor = object()
    store.backup_host_cache_ptrs = object()
    store.backup_src_block_stride = 4
    store.backup_src_block_size = 5
    store.backup_src_rows = 6
    store.backup_row_value_bytes = 0
    current_stream = object()
    recorded_streams: list[object] = []
    store.host_write_event = SimpleNamespace(record=recorded_streams.append)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        torch.accelerator, "current_stream", lambda device: current_stream
    )
    created_events: list[object] = []
    monkeypatch.setattr(hisparse_store_module.torch, "Event", created_events.append)
    monkeypatch.setattr(
        hisparse_store_module.torch,
        "ops",
        SimpleNamespace(
            _C_cache_ops=SimpleNamespace(
                hisparse_backup_layers=lambda *args: calls.append(args)
            )
        ),
    )

    transfer = SparseKVPageTransfer(
        transfer_id=7,
        destination_block_id=5,
        destination_page_offset=1,
        source_block_ids=(11, 13),
        after_forward=False,
    )
    store._enqueue_transfers([transfer])

    assert len(calls) == 1
    assert calls[0][6:] == (4, 4, 5, 6, 0)
    assert store.spill_src_gpu[0, :4].tolist() == [44, 45, 46, 47]
    assert store.spill_src_gpu[1, :4].tolist() == [52, 53, 54, 55]
    assert store.spill_dst_gpu[:4].tolist() == [44, 45, 46, 47]
    assert store.spill_src_gpu.dtype == torch.int64
    assert store.spill_dst_gpu.dtype == torch.int64
    assert store._completed_transfer_ids == [7]
    assert created_events == []
    assert staging_recorded_streams == [current_stream]
    assert recorded_streams == [current_stream]


def test_hisparse_offload_store_finish_forward_enqueues_deferred_spills(monkeypatch):
    store = object.__new__(HiSparseOffloadStore)
    store.hot_backing = SimpleNamespace(device="cuda:0")
    transfer = object()
    store._post_forward_transfers = [transfer]
    current_stream = object()
    recorded_streams: list[object] = []
    store.host_write_event = SimpleNamespace(record=recorded_streams.append)
    calls: list[list[object]] = []
    store._enqueue_transfers = lambda transfers: calls.append(transfers)
    monkeypatch.setattr(
        torch.accelerator, "current_stream", lambda device: current_stream
    )

    store.finish_forward()

    assert calls == [[transfer]]
    assert store._post_forward_transfers == []
    assert recorded_streams == [current_stream]


def test_hisparse_offload_store_finish_step_counts_each_index_group_once():
    store = object.__new__(HiSparseOffloadStore)
    store._metrics_calls = hisparse_store_module._METRICS_INTERVAL - 1
    store._metrics_last = HiSparseStats()
    store.lru_layers = [
        SimpleNamespace(
            offload=SimpleNamespace(
                _swap_stats=torch.tensor([7, 3]), stats_row_bytes=16
            )
        )
    ]

    assert store.finish_step() == HiSparseStats(7, 3, 48)


def test_hisparse_offload_store_reports_each_completed_transfer_once():
    store = object.__new__(HiSparseOffloadStore)
    store._completed_transfer_ids = [3, 5]

    assert store.take_completed_transfer_ids() == [3, 5]
    assert store.take_completed_transfer_ids() is None


def test_hisparse_offload_store_shutdown_releases_pinned_state(monkeypatch):
    store = object.__new__(HiSparseOffloadStore)
    store.layers = []
    released = False

    def release_pinned_state(layers):
        nonlocal released
        assert layers == []
        released = True

    monkeypatch.setattr(
        hisparse_store_module, "release_pinned_state", release_pinned_state
    )

    store.shutdown()

    assert released


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]
