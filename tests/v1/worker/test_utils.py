# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.v1.worker.utils as worker_utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
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
    )
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        worker_hisparse,
        "invalidate_blocks",
        lambda block_ids, block_size: calls.append(
            ("invalidate", block_ids, block_size)
        ),
    )
    runtime.restore_prefix = lambda output: calls.append(("restore", output))

    runtime.pre_step(scheduler_output)

    assert calls == [
        ("invalidate", [2, 3, 4], 64),
        ("restore", scheduler_output),
    ]
    assert runtime.backup_num_items == 2


def test_hisparse_runtime_launches_fused_backup_on_current_stream(monkeypatch):
    runtime = object.__new__(HiSparseRuntime)
    runtime.backup_num_items = 3
    runtime.hot_backing = SimpleNamespace(device="cuda:0")
    runtime.backup_layer_offsets = object()
    runtime.backup_src_indices_ptrs = object()
    runtime.backup_host_anchor = object()
    runtime.backup_host_cache_ptrs = object()
    runtime.backup_dst_slots = object()
    runtime.backup_src_block_stride = 4
    runtime.backup_src_block_size = 5
    runtime.backup_src_rows = 6
    current_stream = object()
    recorded_streams: list[object] = []
    runtime.host_write_event = SimpleNamespace(record=recorded_streams.append)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        torch.accelerator, "current_stream", lambda device: current_stream
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

    runtime.post_forward()

    assert len(calls) == 1
    assert calls[0][6:] == (3, 4, 5, 6)
    assert recorded_streams == [current_stream]


def test_hisparse_runtime_skips_backup_captured_in_graph(monkeypatch):
    runtime = object.__new__(HiSparseRuntime)
    runtime.backup_num_items = 3
    runtime.hot_backing = SimpleNamespace(device="cuda:0")
    current_stream = object()
    recorded_streams: list[object] = []
    runtime.host_write_event = SimpleNamespace(record=recorded_streams.append)
    runtime.backup_decode_rows = lambda num_items: pytest.fail(
        "captured backup must not relaunch"
    )
    monkeypatch.setattr(
        torch.accelerator, "current_stream", lambda device: current_stream
    )

    runtime.post_forward(backup_in_graph=True)

    assert recorded_streams == [current_stream]


def test_hisparse_runtime_post_step_returns_stats(monkeypatch):
    runtime = object.__new__(HiSparseRuntime)
    stats = HiSparseStats(7, 3, 48)
    monkeypatch.setattr(worker_hisparse, "take_hisparse_stats", lambda: stats)

    assert runtime.post_step() is stats


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
