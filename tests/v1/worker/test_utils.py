# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

import vllm.v1.attention.backends.mla.index_group as index_group_module
import vllm.v1.hisparse.runtime as hisparse_runtime_module
from vllm.config import KVTransferConfig
from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import (
    worker as hisparse_worker_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.hisparse.types import SparseKVPageTransfer, SparseKVRowMirror
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
    worker = object.__new__(HiSparseConnectorWorker)
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


def test_hisparse_pre_forward_transfer_builds_page_descriptors():
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.kernel_block_size = 2
    worker.pages_per_host_block = 1
    source = torch.empty((3, 2, 4), dtype=torch.uint8)
    destination = torch.empty((6, 4), dtype=torch.uint8)
    worker.resident_caches = (source,)
    worker.host_caches = (destination,)
    worker.cache_handles = [
        SimpleNamespace(runtime=SimpleNamespace(resident_source_index=0))
    ]
    worker._dma_free_descriptors = []
    worker._submit_dma_descriptors = MagicMock()

    worker._enqueue_transfers(
        [SparseKVPageTransfer(7, 2, 0, (1,), after_forward=False)]
    )

    descriptors, count = worker._submit_dma_descriptors.call_args.args
    assert count == 1
    assert descriptors.src[:count].tolist() == [
        source.data_ptr() + source.stride(0) * source.element_size()
    ]
    assert descriptors.dst[:count].tolist() == [destination.data_ptr() + 4 * 4]
    assert descriptors.sizes[:count].tolist() == [2 * 4]
    assert worker._submit_dma_descriptors.call_args.kwargs["transfer_ids"] == (7,)


def test_hisparse_eager_mirror_skips_pre_forward_page_copy():
    worker = object.__new__(HiSparseConnectorWorker)
    worker.cache_handles = [
        SimpleNamespace(runtime=SimpleNamespace(eager_host_mirror=True))
    ]
    worker._record_transfer_completion = MagicMock()
    worker._enqueue_transfers = MagicMock()
    transfers = [SparseKVPageTransfer(7, 2, 0, (1,), after_forward=False)]

    worker._enqueue_pre_forward_transfers(transfers)

    worker._record_transfer_completion.assert_called_once_with(transfers)
    worker._enqueue_transfers.assert_not_called()


def test_hisparse_lazy_mirror_copies_pre_forward_page():
    worker = object.__new__(HiSparseConnectorWorker)
    worker.cache_handles = [
        SimpleNamespace(runtime=SimpleNamespace(eager_host_mirror=False))
    ]
    worker._record_transfer_completion = MagicMock()
    worker._enqueue_transfers = MagicMock()
    transfers = [SparseKVPageTransfer(7, 2, 0, (1,), after_forward=False)]

    worker._enqueue_pre_forward_transfers(transfers)

    worker._enqueue_transfers.assert_called_once_with(transfers)
    worker._record_transfer_completion.assert_not_called()


def test_hisparse_dma_row_mirror_builds_descriptors(monkeypatch):
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.kernel_block_size = 2
    worker.pages_per_host_block = 1
    source = torch.empty((3, 2, 4), dtype=torch.uint8)
    destination = torch.empty((6, 4), dtype=torch.uint8)
    worker.resident_caches = (source,)
    worker.host_caches = (destination,)
    worker.cache_handles = [
        SimpleNamespace(
            runtime=SimpleNamespace(resident_source_index=0), decode_batch=True
        )
    ]
    worker.hot_backing = SimpleNamespace(device=torch.device("cuda:0"))
    worker.dma_stream = MagicMock()
    worker.host_write_event = MagicMock()
    worker._dma_free_descriptors = []
    worker._pending_dma_descriptors = deque()
    worker._pending_transfer_events = deque()
    worker._enqueued_transfer_ids = []
    worker._dma_submitted = False
    current_stream = MagicMock()
    event = MagicMock()
    swap_blocks_batch = MagicMock()
    monkeypatch.setattr(
        hisparse_worker_module, "current_stream", lambda: current_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch, "Event", lambda: event)
    monkeypatch.setattr(
        hisparse_worker_module.ops, "swap_blocks_batch", swap_blocks_batch
    )

    worker._set_row_mirrors((SparseKVRowMirror((3,), 4, 1),))
    worker._enqueue_row_dma(range(1))

    src_ptrs, dst_ptrs, sizes = swap_blocks_batch.call_args.args
    assert src_ptrs.tolist() == [source.data_ptr() + 3 * 4]
    assert dst_ptrs.tolist() == [destination.data_ptr() + 4 * 4]
    assert sizes.tolist() == [4]
    worker.dma_stream.wait_stream.assert_called_once_with(current_stream)
    worker.host_write_event.record.assert_called_once_with(worker.dma_stream)
    event.record.assert_called_once_with(worker.dma_stream)
    assert worker._dma_submitted


def test_hisparse_slot_mapping_wait_is_deferred_until_first_dma(monkeypatch):
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.kernel_block_size = 64
    worker.dma_stream = MagicMock()
    worker._slot_mapping_host = torch.empty(4, dtype=torch.int64)
    worker._slot_mapping_ready_event = MagicMock()
    worker.cache_handles = [
        SimpleNamespace(mirror_slot_mapping=torch.tensor([64, 65, 130, 131]))
    ]
    worker._set_row_mirrors(
        (
            SparseKVRowMirror((10, 20), 64, 2),
            SparseKVRowMirror((12, 22), 130, 2),
        )
    )
    current_stream = MagicMock()
    monkeypatch.setattr(
        hisparse_worker_module, "current_stream", lambda: current_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    worker._enqueue_slot_mapping_copy()

    worker.dma_stream.wait_stream.assert_called_once_with(current_stream)
    worker._slot_mapping_ready_event.record.assert_called_once_with(worker.dma_stream)
    worker._slot_mapping_ready_event.synchronize.assert_not_called()

    worker._materialize_row_mirror_destinations()

    worker._slot_mapping_ready_event.synchronize.assert_called_once_with()
    assert worker._row_mirrors == (
        SparseKVRowMirror((10, 20), 64, 2),
        SparseKVRowMirror((12, 22), 130, 2),
    )
    assert not worker._slot_mapping_pending


def test_hisparse_slot_mapping_uses_runner_request_order(monkeypatch):
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.kernel_block_size = 64
    worker.dma_stream = MagicMock()
    worker._slot_mapping_host = torch.empty(3, dtype=torch.int64)
    worker._slot_mapping_ready_event = MagicMock()
    worker.cache_handles = [
        SimpleNamespace(mirror_slot_mapping=torch.tensor([64, 130, 131]))
    ]
    row_mirrors = {
        "prefill": (SparseKVRowMirror((20,), 130, 2),),
        "decode": (SparseKVRowMirror((10,), 64, 1),),
    }
    worker._set_row_mirrors(
        hisparse_worker_module._flatten_row_mirrors(row_mirrors, ("decode", "prefill"))
    )
    monkeypatch.setattr(hisparse_worker_module, "current_stream", MagicMock())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    worker._enqueue_slot_mapping_copy()
    worker._materialize_row_mirror_destinations()

    assert worker._row_mirrors == (
        SparseKVRowMirror((10,), 64, 1),
        SparseKVRowMirror((20,), 130, 2),
    )


def test_hisparse_row_mirror_count_must_match_forward():
    worker = object.__new__(HiSparseConnectorWorker)
    worker._set_row_mirrors((SparseKVRowMirror((10,), 64, 2),))

    with pytest.raises(RuntimeError, match="2 rows for 3 tokens"):
        worker._require_row_mirrors(3)


def test_hisparse_row_dma_uses_resident_spans():
    worker = object.__new__(HiSparseConnectorWorker)
    source = torch.empty((2, 2, 4), dtype=torch.uint8)
    destination = torch.empty((12, 4), dtype=torch.uint8)
    worker.is_host_writer = True
    worker.kernel_block_size = 2
    worker.resident_caches = (source,)
    worker.host_caches = (destination,)
    worker.cache_handles = [
        SimpleNamespace(
            runtime=SimpleNamespace(resident_source_index=0), decode_batch=True
        )
    ]
    worker._set_row_mirrors(
        (
            SparseKVRowMirror((0,), 4, 2),
            SparseKVRowMirror((2,), 9, 1),
        )
    )
    worker._dma_free_descriptors = []
    worker._submit_dma_descriptors = MagicMock()

    worker._enqueue_row_dma(range(1))

    descriptors, count = worker._submit_dma_descriptors.call_args.args
    assert count == 2
    assert descriptors.src[:count].tolist() == [
        source.data_ptr(),
        source.data_ptr() + 2 * 4,
    ]
    assert descriptors.dst[:count].tolist() == [
        destination.data_ptr() + 4 * 4,
        destination.data_ptr() + 9 * 4,
    ]
    assert descriptors.sizes[:count].tolist() == [2 * 4, 4]


def test_hisparse_prefill_row_dma_uses_staging_spans():
    worker = object.__new__(HiSparseConnectorWorker)
    resident = torch.empty((2, 2, 4), dtype=torch.uint8)
    staging = torch.empty((2, 2, 4), dtype=torch.uint8)
    destination = torch.empty((12, 4), dtype=torch.uint8)
    worker.is_host_writer = True
    worker.kernel_block_size = 2
    worker.resident_caches = (resident,)
    worker.mirror_caches = (staging,)
    worker.host_caches = (destination,)
    worker.cache_handles = [
        SimpleNamespace(
            runtime=SimpleNamespace(resident_source_index=0), decode_batch=False
        )
    ]
    worker._set_row_mirrors(
        (
            SparseKVRowMirror((2,), 4, 2),
            SparseKVRowMirror((0,), 9, 1),
        )
    )
    worker._dma_free_descriptors = []
    worker._submit_dma_descriptors = MagicMock()

    worker._enqueue_row_dma(range(1))

    descriptors, count = worker._submit_dma_descriptors.call_args.args
    assert count == 2
    assert descriptors.src[:count].tolist() == [
        staging.data_ptr(),
        staging.data_ptr() + 2 * 4,
    ]


def test_hisparse_finish_forward_mirrors_all_layers_once(monkeypatch):
    dst_slots = torch.tensor([7, 8, 9], dtype=torch.int64)
    req_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    leader = SimpleNamespace(
        eager_host_mirror=True,
        is_group_leader=True,
        invalidate_written_slots=MagicMock(),
    )
    follower = SimpleNamespace(
        eager_host_mirror=True,
        is_group_leader=False,
        invalidate_written_slots=MagicMock(),
    )
    handles = [
        SimpleNamespace(
            runtime=runtime,
            decode_batch=True,
            host_mirror_required=True,
            num_actual_tokens=3,
            num_decode_tokens=2,
            req_id_per_token=req_ids,
            mirror_slot_mapping=dst_slots,
        )
        for runtime in (leader, follower)
    ]
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker.hot_backing = SimpleNamespace(device=torch.device("cuda:0"))
    worker._set_row_mirrors((SparseKVRowMirror((0, 0), 7, 3),))
    worker._dma_submitted = False
    worker._per_layer_mirrored = set()
    worker._post_forward_transfers = []
    worker._enqueue_row_dma = MagicMock()
    worker.host_write_event = MagicMock()
    worker._forward_ready_event = MagicMock()
    current_stream = MagicMock()
    monkeypatch.setattr(
        hisparse_worker_module, "current_stream", lambda: current_stream
    )

    worker.finish_forward()

    worker._forward_ready_event.record.assert_called_once_with()
    worker._enqueue_row_dma.assert_called_once_with(
        range(2), ready_event=worker._forward_ready_event
    )
    leader.invalidate_written_slots.assert_called_once()
    torch.testing.assert_close(
        leader.invalidate_written_slots.call_args.args[0], dst_slots[:2]
    )
    follower.invalidate_written_slots.assert_not_called()
    worker.host_write_event.record.assert_called_once_with(current_stream)


def test_hisparse_finish_forward_does_not_repeat_per_layer_mirrors():
    slots = torch.tensor([7, 8], dtype=torch.int64)
    runtime = SimpleNamespace(
        eager_host_mirror=False,
        is_group_leader=False,
        invalidate_written_slots=MagicMock(),
    )
    handles = [
        SimpleNamespace(
            runtime=runtime,
            decode_batch=False,
            host_mirror_required=True,
            num_actual_tokens=2,
            num_decode_tokens=0,
            req_id_per_token=torch.empty(0, dtype=torch.int32),
            mirror_slot_mapping=slots,
        )
        for _ in range(2)
    ]
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker._set_row_mirrors((SparseKVRowMirror((0, 0), 7, 2),))
    worker._per_layer_mirrored = {0, 1}
    worker._enqueue_row_dma = MagicMock()

    worker._enqueue_host_mirror()

    worker._enqueue_row_dma.assert_not_called()


def test_hisparse_lazy_mirror_copies_post_forward_page(monkeypatch):
    runtime = SimpleNamespace(eager_host_mirror=False)
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.cache_handles = [SimpleNamespace(runtime=runtime)]
    transfer = SparseKVPageTransfer(7, 2, 0, (1,), after_forward=True)
    worker._post_forward_transfers = [transfer]
    worker._forward_ready_event = MagicMock()
    worker._enqueue_host_mirror = MagicMock()
    worker._enqueue_transfers = MagicMock()
    worker._record_transfer_completion = MagicMock()
    worker._dma_submitted = True
    monkeypatch.setattr(hisparse_worker_module, "current_stream", MagicMock())

    worker.finish_forward()

    worker._enqueue_transfers.assert_called_once_with([transfer])
    worker._record_transfer_completion.assert_not_called()


def test_hisparse_prefill_mirrors_source_groups_and_flushes_partial_group():
    slots = torch.tensor([7, 8], dtype=torch.int64)
    source_indices = [0, 0, 1, 1, 1, 1, 2]
    handles = [
        SimpleNamespace(
            runtime=SimpleNamespace(
                eager_host_mirror=False,
                is_group_leader=False,
                resident_source_index=source_index,
            ),
            decode_batch=False,
            host_mirror_required=layer_index < 5,
            num_actual_tokens=2 if layer_index < 5 else 0,
            num_decode_tokens=0,
            req_id_per_token=torch.empty(0, dtype=torch.int32),
            mirror_slot_mapping=slots,
        )
        for layer_index, source_index in enumerate(source_indices)
    ]
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker._set_row_mirrors((SparseKVRowMirror((0, 0, 0), 7, 2),))
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = 0
    worker._layer_ready_events = tuple(MagicMock() for _ in handles)
    worker._enqueue_row_dma = MagicMock()

    for layer_index in range(5):
        worker._enqueue_layer_mirror(layer_index)

    worker._enqueue_row_dma.assert_called_once_with(
        range(2), ready_event=worker._layer_ready_events[1]
    )
    worker._enqueue_host_mirror(ready_event=worker._layer_ready_events[-1])
    assert worker._enqueue_row_dma.call_args_list[1].args == (range(2, 5),)
    assert worker._enqueue_row_dma.call_args_list[1].kwargs == {
        "ready_event": worker._layer_ready_events[-1]
    }


def test_hisparse_prefill_mirrors_complete_source_groups():
    slots = torch.tensor([7, 8], dtype=torch.int64)
    source_indices = [0, 0, 1, 1, 1, 1, 2]
    handles = [
        SimpleNamespace(
            runtime=SimpleNamespace(
                eager_host_mirror=False,
                is_group_leader=False,
                resident_source_index=source_index,
            ),
            decode_batch=False,
            host_mirror_required=True,
            num_actual_tokens=2,
            num_decode_tokens=0,
            req_id_per_token=torch.empty(0, dtype=torch.int32),
            mirror_slot_mapping=slots,
        )
        for source_index in source_indices
    ]
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker._set_row_mirrors((SparseKVRowMirror((0, 0, 0), 7, 2),))
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = 0
    worker._layer_ready_events = tuple(MagicMock() for _ in handles)
    worker._enqueue_row_dma = MagicMock()

    for layer_index in range(len(handles)):
        worker._enqueue_layer_mirror(layer_index)

    assert [call.args[0] for call in worker._enqueue_row_dma.call_args_list] == [
        range(2),
        range(2, 6),
        range(6, 7),
    ]


def test_hisparse_finish_forward_rejects_partial_per_layer_mirror():
    slots = torch.tensor([7, 8], dtype=torch.int64)
    runtime = SimpleNamespace(
        eager_host_mirror=False,
        is_group_leader=False,
        invalidate_written_slots=MagicMock(),
    )
    handles = [
        SimpleNamespace(
            runtime=runtime,
            decode_batch=False,
            host_mirror_required=True,
            num_actual_tokens=2,
            num_decode_tokens=0,
            req_id_per_token=torch.empty(0, dtype=torch.int32),
            mirror_slot_mapping=slots,
        )
        for _ in range(2)
    ]
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.cache_handles = handles
    worker._set_row_mirrors((SparseKVRowMirror((0, 0), 7, 2),))
    worker._per_layer_mirrored = {0}
    worker._enqueue_row_dma = MagicMock()

    with pytest.raises(RuntimeError, match="did not mirror every active layer"):
        worker._enqueue_host_mirror()


def test_hisparse_finish_step_orders_next_forward_after_dma(monkeypatch):
    current_stream = MagicMock()
    worker = object.__new__(HiSparseConnectorWorker)
    worker.hot_backing = SimpleNamespace(device=torch.device("cuda:0"))
    worker.is_host_writer = True
    worker._dma_submitted = True
    worker.host_write_event = MagicMock()
    worker._metrics_pending = False
    worker._metrics_calls = 0
    worker._metrics_event = MagicMock()
    worker.leader_runtimes = []
    worker._pending_dma_descriptors = deque()
    worker._dma_free_descriptors = []
    monkeypatch.setattr(
        hisparse_worker_module, "current_stream", lambda: current_stream
    )

    worker.finish_step()

    current_stream.wait_event.assert_called_once_with(worker.host_write_event)
    assert not worker._dma_submitted


def test_hisparse_finish_forward_excludes_trailing_mtp_cache(monkeypatch):
    dst_slots = torch.tensor([7, 8], dtype=torch.int64)
    runtime = SimpleNamespace(
        eager_host_mirror=True,
        is_group_leader=False,
        invalidate_written_slots=MagicMock(),
    )
    active = SimpleNamespace(
        runtime=runtime,
        decode_batch=True,
        host_mirror_required=True,
        num_actual_tokens=2,
        num_decode_tokens=2,
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        mirror_slot_mapping=dst_slots,
    )
    mtp = SimpleNamespace(
        runtime=runtime,
        decode_batch=False,
        host_mirror_required=False,
        num_actual_tokens=0,
        num_decode_tokens=0,
        req_id_per_token=None,
        mirror_slot_mapping=dst_slots,
    )
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.cache_handles = [active, mtp]
    worker._set_row_mirrors((SparseKVRowMirror((0,), 7, 2),))
    worker._per_layer_mirrored = set()
    worker._enqueue_row_dma = MagicMock()

    worker._enqueue_host_mirror()

    worker._enqueue_row_dma.assert_called_once_with(range(1), ready_event=None)


def test_hisparse_shared_host_reader_skips_mirror(monkeypatch):
    """A non-writer TP rank must not mirror rows into the shared host pool."""
    dst_slots = torch.tensor([7, 8], dtype=torch.int64)
    leader = SimpleNamespace(
        eager_host_mirror=True,
        is_group_leader=True,
        invalidate_written_slots=MagicMock(),
    )
    handle = SimpleNamespace(
        runtime=leader,
        decode_batch=True,
        host_mirror_required=True,
        num_actual_tokens=2,
        num_decode_tokens=1,
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
        mirror_slot_mapping=dst_slots,
    )
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = False
    worker.cache_handles = [handle]
    worker._set_row_mirrors((SparseKVRowMirror((0,), 7, 2),))
    worker._per_layer_mirrored = set()
    worker._enqueue_row_dma = MagicMock()

    worker._enqueue_host_mirror()

    worker._enqueue_row_dma.assert_not_called()
    leader.invalidate_written_slots.assert_called_once()


def test_hisparse_shared_host_reader_skips_transfer_completion():
    """A non-writer TP rank must not acknowledge host writes."""
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = False
    worker.kernel_block_size = 1
    worker._enqueued_transfer_ids = []
    worker._pending_transfer_events = []

    worker._record_transfer_completion([SparseKVPageTransfer(1, 2, 0, (3,), True)])

    assert worker._enqueued_transfer_ids == []
    assert worker._pending_transfer_events == []


def test_hisparse_writer_records_transfer_completion_after_dma(monkeypatch):
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.dma_stream = MagicMock()
    worker._enqueued_transfer_ids = []
    worker._pending_transfer_events = deque()
    event = MagicMock()
    monkeypatch.setattr(torch, "Event", lambda: event)

    worker._record_transfer_completion(
        [
            SparseKVPageTransfer(3, 2, 0, (1,), False),
            SparseKVPageTransfer(7, 4, 0, (5,), True),
        ]
    )

    event.record.assert_called_once_with(worker.dma_stream)
    assert worker._enqueued_transfer_ids == [3, 7]
    assert worker._pending_transfer_events == deque([(event, (3, 7))])


@pytest.mark.parametrize("is_host_writer", [False, True])
def test_hisparse_step_waits_for_previous_host_write(monkeypatch, is_host_writer):
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = is_host_writer
    worker.hot_backing = SimpleNamespace(device=torch.device("cuda:1"))
    host_write_events = (MagicMock(), MagicMock())
    worker.host_write_events = host_write_events
    worker.host_write_event = host_write_events[1]
    worker._next_host_write_event = 0
    worker.host_caches = ()
    worker.host_num_blocks = 1
    worker._post_forward_transfers = []
    worker._pending_invalid_block_ids = []
    worker.cache_handles = []
    worker._per_layer_mirrored = set()
    worker._layer_mirror_callbacks = ()
    stream = MagicMock()
    monkeypatch.setattr(hisparse_worker_module, "current_stream", lambda: stream)

    worker.start_step(
        SimpleNamespace(
            host_block_copies=[], command=None, source_block_ids=[], row_mirrors={}
        ),
        None,
    )
    worker.start_step(
        SimpleNamespace(
            host_block_copies=[], command=None, source_block_ids=[], row_mirrors={}
        ),
        None,
    )

    assert stream.wait_event.call_args_list == [
        call(host_write_events[1]),
        call(host_write_events[0]),
    ]
    assert worker.host_write_event is host_write_events[1]


@pytest.mark.parametrize("tp_rank, expected_copies", [(0, 1), (1, 0)])
def test_hisparse_shared_host_block_copy_has_one_writer(
    monkeypatch, tp_rank, expected_copies
):
    worker = object.__new__(HiSparseConnectorWorker)
    worker.shared_host_region = object()
    worker.host_caches = (torch.empty(4, 1),)
    worker.host_num_blocks = 4
    previous_event = MagicMock()
    copies = (KVCacheBlockCopy(0, 1, None),)
    copy_blocks = MagicMock()
    tp_group = MagicMock()
    monkeypatch.setattr(
        hisparse_worker_module, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(hisparse_worker_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        hisparse_worker_module, "copy_kv_cache_blocks_inplace", copy_blocks
    )

    worker._copy_host_blocks(copies, previous_event)

    assert copy_blocks.call_count == expected_copies
    tp_group.barrier.assert_called_once_with()


def test_hisparse_empty_step_does_not_replay_stale_host_mirror(monkeypatch):
    handle = SimpleNamespace(
        runtime=SimpleNamespace(eager_host_mirror=True),
        decode_batch=True,
        num_actual_tokens=2,
        num_decode_tokens=2,
        req_id_per_token=torch.tensor([0, 1]),
        mirror_slot_mapping=torch.tensor([4, 5]),
    )
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.hot_backing = SimpleNamespace(device=torch.device("cpu"))
    worker.host_write_event = MagicMock()
    worker._forward_ready_event = MagicMock()
    worker.host_caches = ()
    worker.host_num_blocks = 1
    worker.cache_handles = [handle]
    worker._per_layer_mirrored = set()
    worker._layer_mirror_callbacks = (MagicMock(),)
    worker._post_forward_transfers = []
    worker._pending_invalid_block_ids = []
    worker._enqueue_host_mirror = MagicMock(wraps=worker._enqueue_host_mirror)
    stream = MagicMock()
    monkeypatch.setattr(hisparse_worker_module, "current_stream", lambda: stream)

    worker.start_step(
        SimpleNamespace(
            host_block_copies=[], command=None, source_block_ids=[], row_mirrors={}
        ),
        None,
    )
    worker.finish_forward()

    worker._enqueue_host_mirror.assert_called_once_with(worker._forward_ready_event)
    assert handle.num_actual_tokens == 0
    torch.testing.assert_close(handle.mirror_slot_mapping, torch.tensor([4, 5]))


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
    shared_states: list[object] = []
    streams: list[object] = []

    def create_shared_state(_device, _max_rows, _top_k):
        shared_states.append(object())
        return shared_states[-1]

    def create_stream(_device):
        streams.append(object())
        return streams[-1]

    monkeypatch.setattr(
        hisparse_runtime_module, "_create_shared_topk_state", create_shared_state
    )
    monkeypatch.setattr(hisparse_runtime_module, "_create_copy_stream", create_stream)
    monkeypatch.setattr(index_group_module, "_create_side_stream", lambda _: object())
    monkeypatch.setattr(index_group_module, "_create_event", lambda: object())
    index_group_builder = index_group_module.SparseMLAIndexGroupBuilder(
        torch.empty((2, 4), dtype=torch.int32)
    )

    def make_cache_handle(is_leader: bool):
        index_group, _ = index_group_builder.register_layer(is_leader)
        cache_handle = hisparse_runtime_module.create_hisparse_cache_handle(
            config,
            model_top_k=4,
            is_index_group_leader=is_leader,
            row_width=8,
            kv_dtype=torch.float32,
            index_group=index_group,
            device="cpu",
        )
        assert cache_handle is not None
        return cache_handle

    first_leader = make_cache_handle(True)
    first_index_group = index_group_builder.current_group
    first_follower = make_cache_handle(False)
    second_leader = make_cache_handle(True)
    second_index_group = index_group_builder.current_group
    second_follower = make_cache_handle(False)

    assert first_index_group is not None
    assert second_index_group is not None
    assert first_follower.runtime.index_group is first_leader.runtime.index_group
    assert second_follower.runtime.index_group is second_leader.runtime.index_group
    assert first_leader.runtime.index_group is not second_leader.runtime.index_group
    assert first_leader.runtime.index_group.copy_stream is first_index_group.side_stream
    assert (
        first_leader.runtime.index_group.logical_topk_ready
        is first_index_group.logical_topk_ready
    )
    assert (
        second_leader.runtime.index_group.copy_stream is second_index_group.side_stream
    )
    assert (
        second_leader.runtime.index_group.logical_topk_ready
        is second_index_group.logical_topk_ready
    )
    assert len(shared_states) == 2
    assert streams == []


@pytest.mark.parametrize(
    "kv_transfer_config",
    [None, KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both")],
)
def test_hisparse_cache_eagerly_mirrors_host_rows(monkeypatch, kv_transfer_config):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=2,
        ),
        speculative_config=None,
        kv_transfer_config=kv_transfer_config,
    )
    resolved = hisparse_runtime_module.ResolvedHiSparseConfig(
        top_k=4,
        device_buffer_size=8,
        host_pool_gib=1.0,
    )
    monkeypatch.setattr(
        hisparse_runtime_module.ResolvedHiSparseConfig,
        "from_vllm_config",
        classmethod(lambda cls, vllm_config, model_top_k: resolved),
    )
    runtime = SimpleNamespace(index_group=object(), eager_host_mirror=True)
    monkeypatch.setattr(
        hisparse_runtime_module, "HiSparseRuntime", lambda **kwargs: runtime
    )

    cache_handle = hisparse_runtime_module.create_hisparse_cache_handle(
        config,
        model_top_k=4,
        is_index_group_leader=True,
        row_width=8,
        kv_dtype=torch.float32,
        device="cpu",
    )

    assert cache_handle is not None
    assert cache_handle.runtime.eager_host_mirror


@pytest.mark.parametrize("eager_host_mirror", [True, False])
def test_hisparse_runtime_takes_eager_host_mirror_from_config(
    monkeypatch, eager_host_mirror
):
    monkeypatch.setattr(hisparse_runtime_module, "_has_hisparse_ops", lambda: True)
    runtime = hisparse_runtime_module.HiSparseRuntime(
        config=hisparse_runtime_module.ResolvedHiSparseConfig(
            top_k=4,
            device_buffer_size=8,
            host_pool_gib=1.0,
            eager_host_mirror=eager_host_mirror,
        ),
        max_num_reqs=2,
        row_width=8,
        kv_dtype=torch.float32,
        device="cpu",
        index_group=SimpleNamespace(followers=[]),
    )

    assert runtime.eager_host_mirror is eager_host_mirror


class _TestReplaySSMMixer(MambaMixer2):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.use_replayssm = True
        self.mamba_config = MambaConfig(backend=MambaBackendEnum.FLASHINFER)
        self._replayssm_ring_start = torch.empty(0, dtype=torch.int32)
        self._replayssm_prev_num_accepted = torch.empty(0, dtype=torch.int32)
        self._updates_replayssm_trackers = True

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((2,), (3,), (4,), (5,), (6,))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.float32,) * 5


def _packed_replayssm_cache(num_blocks: int) -> torch.Tensor:
    return torch.full((num_blocks, 1, 1, 80), 0, dtype=torch.int8)


def test_bind_kv_cache_shares_replayssm_trackers_by_cache_group():
    mixers = [_TestReplaySSMMixer() for _ in range(3)]
    layer_names = [f"layers.{i}.mixer" for i in range(3)]
    ctx = dict(zip(layer_names, mixers))
    # Reverse insertion order: updater must follow layer index, not dict order.
    kv_cache = {
        layer_names[2]: _packed_replayssm_cache(4),
        layer_names[1]: _packed_replayssm_cache(4),
        layer_names[0]: _packed_replayssm_cache(4),
    }
    kv_cache_groups = [
        SimpleNamespace(layer_names=[layer_names[0], layer_names[2]]),
        SimpleNamespace(layer_names=[layer_names[1]]),
    ]

    bind_kv_cache(kv_cache, ctx, [], kv_cache_groups=kv_cache_groups)

    assert (
        mixers[0]._replayssm_ring_start.data_ptr()
        == mixers[2]._replayssm_ring_start.data_ptr()
    )
    assert (
        mixers[0]._replayssm_prev_num_accepted.data_ptr()
        == mixers[2]._replayssm_prev_num_accepted.data_ptr()
    )
    assert (
        mixers[1]._replayssm_ring_start.data_ptr()
        != mixers[0]._replayssm_ring_start.data_ptr()
    )
    # Group {0, 2} shares trackers; layer 2 (not 0) updates after both run.
    assert [m._updates_replayssm_trackers for m in mixers] == [False, True, True]


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
