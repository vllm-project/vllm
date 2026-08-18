# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace
from typing import Any

import torch

import vllm.v1.kv_offload.sparse.hisparse_runtime as hisparse_runtime_module
import vllm.v1.kv_offload.sparse.hisparse_worker as hisparse_worker_module
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_offload.sparse.base import (
    SparseKVPageTransfer,
)
from vllm.v1.kv_offload.sparse.hisparse_worker import (
    HiSparseWorker,
)
from vllm.v1.worker.utils import bind_kv_cache, copy_kv_cache_blocks_inplace


def test_copy_cpu_kv_cache_logical_blocks_ignores_storage_padding():
    waited_for_host_writes = False

    def wait_for_host_writes():
        nonlocal waited_for_host_writes
        waited_for_host_writes = True

    host_write_event = SimpleNamespace(synchronize=wait_for_host_writes)
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
        host_write_event=host_write_event,
    )

    torch.testing.assert_close(cache[0:2], torch.full_like(cache[0:2], 7))
    torch.testing.assert_close(cache[4:6], torch.full_like(cache[4:6], 11))
    assert waited_for_host_writes
    assert (backing[0] == -1).all()
    assert (backing[9] == -1).all()


def test_hisparse_worker_updates_request_state_mapping_in_place(monkeypatch):
    worker = object.__new__(HiSparseWorker)
    worker.request_state_indices = torch.arange(4, dtype=torch.int32)
    worker._pending_invalid_block_ids = [5]
    invalidations = []
    worker.invalidate_blocks = lambda blocks, states: invalidations.append(
        (blocks.copy(), states.clone())
    )
    original_ptr = worker.request_state_indices.data_ptr()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    worker.set_request_state_indices(torch.tensor([3, 1], dtype=torch.int32))

    assert worker.request_state_indices.data_ptr() == original_ptr
    assert worker.request_state_indices.tolist() == [3, 1, -1, -1]
    assert len(invalidations) == 1
    assert invalidations[0][0] == [5]
    torch.testing.assert_close(
        invalidations[0][1], torch.tensor([3, 1], dtype=torch.int32)
    )
    assert worker._pending_invalid_block_ids == []


def test_hisparse_runtime_invalidates_only_scheduled_request_states():
    runtime = object.__new__(hisparse_runtime_module.HiSparseRuntime)
    runtime.device = torch.device("cpu")
    runtime.index_group = SimpleNamespace(
        device_global_indices=torch.tensor(
            [[6, 7, 8], [6, 9, 10], [6, 11, 12]], dtype=torch.int32
        )
    )

    runtime.invalidate_slots(torch.tensor([6]), torch.tensor([1]))

    torch.testing.assert_close(
        runtime.index_group.device_global_indices,
        torch.tensor([[6, 7, 8], [-1, 9, 10], [6, 11, 12]], dtype=torch.int32),
    )


def test_hisparse_cache_handles_join_index_groups_during_construction(monkeypatch):
    """Followers must not allocate duplicate runtime state before profiling."""
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=2,
        ),
        speculative_config=None,
        kv_transfer_config=None,
    )
    resolved = hisparse_runtime_module.ResolvedHiSparseConfig(
        top_k=4,
        device_buffer_size=8,
        host_pool_gib=1.0,
    )
    monkeypatch.setattr(hisparse_runtime_module, "_has_hisparse_ops", lambda: True)
    monkeypatch.setattr(
        hisparse_runtime_module.ResolvedHiSparseConfig,
        "from_vllm_config",
        classmethod(lambda cls, vllm_config, model_top_k: resolved),
    )
    plans: list[object] = []
    streams: list[object] = []

    def create_plan(_device, _max_rows, _top_k):
        plans.append(object())
        return plans[-1]

    def create_stream(_device):
        streams.append(object())
        return streams[-1]

    monkeypatch.setattr(hisparse_runtime_module, "_create_group_plan", create_plan)
    monkeypatch.setattr(hisparse_runtime_module, "_create_copy_stream", create_stream)
    index_group_builder = hisparse_runtime_module.HiSparseIndexGroupBuilder()

    def make_cache_handle(is_leader: bool):
        cache_handle = hisparse_runtime_module.create_hisparse_cache_handle(
            config,
            model_top_k=4,
            is_index_group_leader=is_leader,
            row_width=8,
            kv_dtype=torch.float32,
            index_group_builder=index_group_builder,
            device="cpu",
        )
        assert cache_handle is not None
        return cache_handle

    first_leader = make_cache_handle(True)
    first_follower = make_cache_handle(False)
    second_leader = make_cache_handle(True)
    second_follower = make_cache_handle(False)

    assert first_follower.runtime.index_group is first_leader.runtime.index_group
    assert second_follower.runtime.index_group is second_leader.runtime.index_group
    assert first_leader.runtime.index_group is not second_leader.runtime.index_group
    assert first_leader.runtime.index_group.leader is first_leader.runtime
    assert second_leader.runtime.index_group.leader is second_leader.runtime
    assert len(plans) == len(streams) == 2
    assert not first_follower.runtime.is_group_leader
    assert not second_follower.runtime.is_group_leader


def test_hisparse_worker_preserves_directly_imported_indexer(monkeypatch):
    worker = object.__new__(HiSparseWorker)
    worker.kernel_block_size = 4
    worker.blocks_per_kv_block = 2
    worker.source_group_id = 2
    worker.indexer_group_id = 0
    worker.src_cpu = torch.empty(2, dtype=torch.int32)
    worker.dst_cpu = torch.empty(2, dtype=torch.int32)
    worker.src_gpu = torch.empty(2, dtype=torch.int32)
    worker.dst_gpu = torch.empty(2, dtype=torch.int32)
    worker.cache_pairs = [(torch.empty(1), torch.empty(1))]
    copied: list[tuple[list[int], list[int]]] = []

    def copy_blocks(source, indexer, src, dst):
        copied.append((src.tolist(), dst.tolist()))

    monkeypatch.setattr(torch.ops._C_cache_ops, "hisparse_copy_blocks", copy_blocks)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="direct", block_ids=([10], [], [1]), num_computed_tokens=4
            ),
            SimpleNamespace(
                req_id="host", block_ids=([20], [], [2]), num_computed_tokens=4
            ),
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            resumed_req_ids=set(),
        ),
    )

    worker.restore_prefix(scheduler_output, {"direct"})

    assert copied == [([4], [20])]


def test_hisparse_worker_prepare_step_accepts_warmup_without_command():
    worker = object.__new__(HiSparseWorker)
    worker.kernel_block_size = 64
    worker._pending_invalid_block_ids = []
    worker._post_forward_transfers = [object()]
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
    )
    cache_handle = SimpleNamespace(fully_resident=True)
    worker.cache_handles = [cache_handle]
    calls = []
    worker.restore_prefix = lambda output, ready: calls.append(
        ("restore", output, ready)
    )

    worker.prepare_step(None, scheduler_output)

    assert not cache_handle.fully_resident
    assert worker._post_forward_transfers == []
    assert calls == [("restore", scheduler_output, ())]
    assert worker._pending_invalid_block_ids == []


def test_hisparse_worker_enqueues_fused_page_spill(monkeypatch):
    worker = object.__new__(HiSparseWorker)
    worker.kernel_block_size = 4
    worker.blocks_per_kv_block = 2
    worker.spill_row_capacity = 8
    worker.spill_src_cpu = torch.empty((1, 2, 8), dtype=torch.int64)
    worker.spill_dst_cpu = torch.empty((1, 8), dtype=torch.int64)
    worker.spill_src_gpu = torch.empty((2, 8), dtype=torch.int64)
    worker.spill_dst_gpu = torch.empty(8, dtype=torch.int64)
    worker._spill_staging_index = 0
    staging_recorded_streams: list[object] = []
    worker._spill_staging_events = [
        SimpleNamespace(query=lambda: True, record=staging_recorded_streams.append)
    ]
    worker.spill_src_indices_ptrs = object()
    worker.cache_handles = [
        SimpleNamespace(runtime=SimpleNamespace(resident_source_index=0)),
        SimpleNamespace(runtime=SimpleNamespace(resident_source_index=1)),
    ]
    worker._enqueued_transfer_ids = []
    worker._pending_transfer_events = deque()
    worker.hot_backing = SimpleNamespace(device="cuda:0")
    worker.backup_layer_offsets = object()
    worker.backup_host_anchor = object()
    worker.backup_host_cache_ptrs = object()
    worker.backup_src_block_stride = 4
    worker.backup_src_block_size = 5
    worker.backup_src_rows = 6
    worker.backup_row_value_bytes = 0
    current_stream = object()
    recorded_streams: list[object] = []
    worker.host_write_event = SimpleNamespace(record=recorded_streams.append)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        torch.accelerator, "current_stream", lambda device: current_stream
    )
    completion_recorded_streams: list[object] = []
    completion_event = SimpleNamespace(
        query=lambda: True, record=completion_recorded_streams.append
    )
    monkeypatch.setattr(hisparse_worker_module.torch, "Event", lambda: completion_event)
    monkeypatch.setattr(
        hisparse_worker_module.torch,
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
    worker._enqueue_transfers([transfer])

    assert len(calls) == 1
    assert calls[0][6:] == (4, 4, 5, 6, 0)
    assert worker.spill_src_gpu[0, :4].tolist() == [44, 45, 46, 47]
    assert worker.spill_src_gpu[1, :4].tolist() == [52, 53, 54, 55]
    assert worker.spill_dst_gpu[:4].tolist() == [44, 45, 46, 47]
    assert worker.spill_src_gpu.dtype == torch.int64
    assert worker.spill_dst_gpu.dtype == torch.int64
    assert worker._enqueued_transfer_ids == [7]
    assert list(worker._pending_transfer_events) == [(completion_event, (7,))]
    assert staging_recorded_streams == [current_stream]
    assert recorded_streams == [current_stream]
    assert completion_recorded_streams == [current_stream]


def test_hisparse_worker_reports_each_completed_transfer_once():
    worker = object.__new__(HiSparseWorker)
    worker._enqueued_transfer_ids = [3, 5]
    queries = iter((False, True))
    event = SimpleNamespace(query=lambda: next(queries))
    worker._pending_transfer_events = deque([(event, (3, 5))])

    assert worker.take_transfer_updates() == ([3, 5], [])
    assert worker.take_transfer_updates() == ([], [3, 5])
    assert worker.take_transfer_updates() == ([], [])


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
