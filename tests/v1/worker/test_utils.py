# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import torch

import vllm.v1.worker.utils as worker_utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import HiSparseSpill
from vllm.v1.metrics.stats import HiSparseStats
from vllm.v1.worker.gpu import hisparse as worker_hisparse
from vllm.v1.worker.gpu.hisparse import HiSparseRuntime, _expand_source_block_ids
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


def test_hisparse_runtime_pre_step_invalidates_and_restores(monkeypatch):
    runtime = object.__new__(HiSparseRuntime)
    runtime.block_size = 64
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-0": 1, "request-1": 1},
        scheduled_new_reqs=[SimpleNamespace(block_ids=([2, 3],))],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[([4],), None]),
        hisparse_spills=None,
        hisparse_fully_resident=True,
    )
    runtime.coordinators = []
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        worker_hisparse,
        "invalidate_blocks",
        lambda block_ids, block_size: calls.append(
            ("invalidate", block_ids, block_size)
        ),
    )
    runtime.restore_prefix = lambda output: calls.append(("restore", output))
    runtime._enqueue_spills = lambda spills: calls.append(("spill", spills))

    runtime.pre_step(scheduler_output)

    assert calls == [
        ("spill", []),
        ("invalidate", [2, 3, 4], 64),
        ("restore", scheduler_output),
    ]
    assert runtime.fully_resident_batch
    assert runtime._post_forward_spills == []


def test_hisparse_runtime_enqueues_fused_page_spill(monkeypatch):
    runtime = object.__new__(HiSparseRuntime)
    runtime.kernel_block_size = 4
    runtime.blocks_per_kv_block = 2
    runtime.spill_row_capacity = 8
    runtime.spill_staging_count = 1
    runtime.spill_src_cpu = torch.empty((1, 2, 8), dtype=torch.int64)
    runtime.spill_dst_cpu = torch.empty((1, 8), dtype=torch.int64)
    runtime.spill_src_gpu = torch.empty((2, 8), dtype=torch.int64)
    runtime.spill_dst_gpu = torch.empty(8, dtype=torch.int64)
    runtime._spill_staging_index = 0
    runtime._spill_staging_events = [None]
    runtime.spill_src_indices_ptrs = object()
    runtime.coordinators = [
        SimpleNamespace(resident_group_id=2),
        SimpleNamespace(resident_group_id=3),
    ]
    runtime._enqueued_spill_ids = []
    runtime.hot_backing = SimpleNamespace(device="cuda:0")
    runtime.backup_layer_offsets = object()
    runtime.backup_host_anchor = object()
    runtime.backup_host_cache_ptrs = object()
    runtime.backup_src_block_stride = 4
    runtime.backup_src_block_size = 5
    runtime.backup_src_rows = 6
    current_stream = object()
    recorded_streams: list[object] = []
    staging_recorded_streams: list[object] = []
    runtime.host_write_event = SimpleNamespace(record=recorded_streams.append)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        torch.accelerator, "current_stream", lambda device: current_stream
    )
    monkeypatch.setattr(
        worker_hisparse.torch,
        "Event",
        lambda: SimpleNamespace(record=staging_recorded_streams.append),
    )
    monkeypatch.setattr(
        worker_hisparse.torch,
        "ops",
        SimpleNamespace(
            _C_cache_ops=SimpleNamespace(
                hisparse_backup_layers=lambda *args: calls.append(args)
            )
        ),
    )

    spill = HiSparseSpill(
        spill_id=7,
        request_id="request-0",
        page_index=3,
        host_block_id=5,
        host_page_offset=1,
        resident_block_ids=((2, 11), (3, 13)),
        after_forward=False,
    )
    runtime._enqueue_spills([spill])

    assert len(calls) == 1
    assert calls[0][6:] == (4, 4, 5, 6)
    assert runtime.spill_src_gpu[0, :4].tolist() == [44, 45, 46, 47]
    assert runtime.spill_src_gpu[1, :4].tolist() == [52, 53, 54, 55]
    assert runtime.spill_dst_gpu[:4].tolist() == [44, 45, 46, 47]
    assert runtime.spill_src_gpu.dtype == torch.int64
    assert runtime.spill_dst_gpu.dtype == torch.int64
    assert runtime._enqueued_spill_ids == [7]
    assert staging_recorded_streams == [current_stream]
    assert recorded_streams == [current_stream]


def test_hisparse_runtime_post_forward_enqueues_deferred_spills(monkeypatch):
    runtime = object.__new__(HiSparseRuntime)
    runtime.hot_backing = SimpleNamespace(device="cuda:0")
    spill = object()
    runtime._post_forward_spills = [spill]
    current_stream = object()
    recorded_streams: list[object] = []
    runtime.host_write_event = SimpleNamespace(record=recorded_streams.append)
    calls: list[list[object]] = []
    runtime._enqueue_spills = lambda spills: calls.append(spills)
    monkeypatch.setattr(
        torch.accelerator, "current_stream", lambda device: current_stream
    )

    runtime.post_forward()

    assert calls == [[spill]]
    assert runtime._post_forward_spills == []
    assert recorded_streams == [current_stream]


def test_hisparse_runtime_post_step_returns_stats(monkeypatch):
    runtime = object.__new__(HiSparseRuntime)
    stats = HiSparseStats(7, 3, 48)
    monkeypatch.setattr(worker_hisparse, "take_hisparse_stats", lambda: stats)

    assert runtime.post_step() is stats


def test_hisparse_runtime_reports_each_enqueued_spill_once():
    runtime = object.__new__(HiSparseRuntime)
    runtime._enqueued_spill_ids = [3, 5]

    assert runtime.take_spill_completions() == [3, 5]
    assert runtime.take_spill_completions() is None


def test_hisparse_runtime_shutdown_releases_pinned_state(monkeypatch):
    runtime = object.__new__(HiSparseRuntime)
    released = False

    def release_pinned_state():
        nonlocal released
        released = True

    monkeypatch.setattr(worker_hisparse, "release_pinned_state", release_pinned_state)

    runtime.shutdown()

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
