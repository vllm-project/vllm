# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import (
    worker as worker_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
    HiSparseConnectorMetadata,
    HiSparseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.hisparse import runtime as runtime_module
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    update_hisparse_residency,
)
from vllm.v1.hisparse.types import (
    SparseKVOffloadCommand,
    SparseKVPageTransfer,
    SparseKVResidencyUpdate,
    SparseKVRowMirror,
)
from vllm.v1.metrics.stats import KVCacheEvictionEvent
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


@pytest.mark.parametrize("nested", [False, True])
def test_cache_manager_binding_preserves_hisparse_and_legacy_pool_hooks(nested):
    """Composite binding must reach HiSparse and existing pool-only connectors."""
    hisparse = object.__new__(HiSparseConnector)
    hisparse._role = KVConnectorRole.SCHEDULER
    hisparse.connector_scheduler = HiSparseConnectorScheduler(async_speculative=False)
    legacy = object.__new__(ExampleConnector)
    legacy.bind_gpu_block_pool = MagicMock()
    connector = object.__new__(MultiConnector)
    connector._connectors = [hisparse, legacy]
    if nested:
        parent = object.__new__(MultiConnector)
        parent._connectors = [connector]
        connector = parent
    from tests.v1.core.test_prefix_caching import make_hisparse_kv_cache_manager

    manager = make_hisparse_kv_cache_manager(16, 16)

    connector.bind_kv_cache_manager(manager)

    assert hisparse.connector_scheduler.coordinator is get_hisparse_coordinator(manager)
    legacy.bind_gpu_block_pool.assert_called_once_with(manager.block_pool)


def test_hisparse_requires_block_outermost_device_layout():
    assert HiSparseConnector.get_required_kvcache_layout(MagicMock()) == "BLHNC"


def test_scheduler_stats_report_host_pool_usage():
    """The scheduler-side stats sample the coordinator's host pool level."""
    from tests.v1.core.test_prefix_caching import (
        HISPARSE_BLOCK_SIZE,
        make_hisparse_kv_cache_manager,
        make_request,
        sha256,
    )
    from vllm.v1.core.kv_cache_utils import init_none_hash

    init_none_hash(sha256)

    manager = make_hisparse_kv_cache_manager(32, host_num_blocks=8)
    scheduler = HiSparseConnectorScheduler(async_speculative=False)
    scheduler.bind_coordinator(get_hisparse_coordinator(manager))
    coordinator = scheduler.coordinator

    stats = scheduler.get_kv_connector_stats()
    assert stats.data["host_cache_usage_perc"] == [0.0]
    assert stats.data["pending_page_transfers"] == [0]

    request = make_request(
        "request", list(range(4 * HISPARSE_BLOCK_SIZE)), HISPARSE_BLOCK_SIZE, sha256
    )
    assert manager.allocate_slots(request, num_new_tokens=64) is not None
    host_blocks = coordinator.host_manager.req_to_blocks[request.request_id]
    num_used = sum(not block.is_null for block in host_blocks)

    # The pool reserves the null block, so 8 configured blocks yield 7 usable.
    expected = pytest.approx(num_used / 7)
    assert 0 < num_used <= 7
    for _ in range(2):
        stats = scheduler.get_kv_connector_stats()
        assert stats.data["host_cache_usage_perc"] == [expected]


def test_hisparse_host_residency_reported_as_hisparse_metrics():
    """Host blocks sample into HiSparse's own residency metrics."""
    from tests.v1.core.test_prefix_caching import (
        make_hisparse_kv_cache_config,
        make_kv_cache_manager,
    )
    from tests.v1.core.utils import create_requests

    collector = KVCacheMetricsCollector(sample_rate=1.0)
    manager = make_kv_cache_manager(
        make_hisparse_kv_cache_config(2, 2),
        max_model_len=128,
        enable_caching=True,
        hash_block_size=16,
        metrics_collector=collector,
    )
    coordinator = get_hisparse_coordinator(manager)
    assert coordinator.host_manager is not None
    scheduler = HiSparseConnectorScheduler(async_speculative=False)
    scheduler.bind_coordinator(coordinator)
    device = manager.block_pool
    host = coordinator.host_manager.block_pool
    request = create_requests(1, num_tokens=16)[0]
    for pool, birth in ((device, 1), (host, 2)):
        with patch("time.monotonic_ns", return_value=birth * 10**9):
            block = pool.get_new_blocks(1)[0]
            assert block.block_id == 1
            pool.cache_full_blocks(request, [block], 0, 1, 16, 0)
    with patch("time.monotonic_ns", return_value=4_000_000_000):
        host.free_blocks([host.blocks[1]])
        host.evict_blocks({1})
    with patch("time.monotonic_ns", return_value=6_000_000_000):
        device.evict_blocks({1})

    assert collector.drain_events() == [KVCacheEvictionEvent(5.0, 5.0, ())]
    stats = scheduler.get_kv_connector_stats()
    assert stats.data["host_block_lifetime_seconds"] == [2.0]
    assert stats.data["host_block_idle_before_evict_seconds"] == [2.0]


def test_no_forward_enqueues_deferred_hisparse_transfers():
    """A zero-token step must still enqueue deferred post-forward transfers."""
    connector = object.__new__(ActiveKVConnector)
    connector._disabled = False
    connector.pre_forward = MagicMock()
    connector.finish_forward = MagicMock()
    connector.post_forward = MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(finished_req_ids=set())
    connector.no_forward(scheduler_output)

    connector.pre_forward.assert_called_once_with(scheduler_output)
    connector.finish_forward.assert_called_once_with()


def test_full_graph_step_prepares_host_mirror_outside_model():
    """Graph replay must restore host-mirror state cleared at step start."""
    runtime = SimpleNamespace(
        is_group_leader=True,
        eager_host_mirror=True,
        begin_forward=MagicMock(),
        invalidate_written_slots=MagicMock(),
    )
    handle = HiSparseCacheHandle(runtime)
    handle.mirror_slot_mapping = torch.tensor([4, 5])
    worker = object.__new__(HiSparseConnectorWorker)
    worker.cache_layer_names = ["layer"]
    worker.cache_handles = [handle]
    worker._group_leaders = (("layer", handle),)
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker._draft_layers = ()
    worker.is_host_writer = True
    worker._enqueue_row_dma = MagicMock()
    worker.start_step = MagicMock(
        side_effect=lambda *_args, **_kwargs: worker._clear_forward_mirror_state()
    )

    connector = object.__new__(HiSparseConnector)
    connector.connector_worker = worker
    connector._get_connector_metadata = MagicMock(
        return_value=HiSparseConnectorMetadata(None, (), (), {}, True, {})
    )
    req_id_per_token = torch.tensor([0, 1], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decode_tokens=2,
        num_reqs=2,
        max_query_len=1,
        req_id_per_token=req_id_per_token,
    )

    connector.start_load_kv(
        SimpleNamespace(),
        request_state_indices=None,
        request_ids=[],
        attn_metadata={"layer": attn_metadata},
    )

    worker._enqueue_host_mirror()

    worker._enqueue_row_dma.assert_called_once_with((0,), ready_event=None)
    runtime.invalidate_written_slots.assert_called_once()


class _FakeEvent:
    log: list[tuple[str, object]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, stream=None) -> None:
        self.log.append(("event", self))

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


@pytest.mark.parametrize("cg_mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])
def test_draft_layer_rows_mirrored_after_drafter(monkeypatch, cg_mode):
    """Draft-layer host rows must hold the drafter's writes before page release.

    The drafter writes its layers' rows after the target forward. Mirroring
    them at the end of the target forward (the only mirror a FULL graph gets)
    leaves pre-draft rows on the host, and the page is still reported clean.
    """
    block_size, width = 4, 8
    resident_block, host_block, transfer_id = 2, 1, 7
    log: list[tuple[str, object]] = []
    _FakeEvent.log = log

    def swap_blocks_batch(src, dst, sizes):
        for src_ptr, dst_ptr, size in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(dst_ptr, src_ptr, size)
            log.append(("dma", dst_ptr))

    monkeypatch.setattr(torch, "Event", _FakeEvent)
    monkeypatch.setattr(torch, "Stream", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: contextlib.nullcontext())
    compute_stream = MagicMock()
    monkeypatch.setattr(worker_module, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(worker_module.ops, "swap_blocks_batch", swap_blocks_batch)
    monkeypatch.setattr(
        runtime_module,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=cg_mode),
    )

    slots = torch.arange(block_size) + resident_block * block_size
    request_state_indices = torch.full((1,), -1, dtype=torch.int32)
    handles = []
    for draft_layer in (False, True):
        runtime = SimpleNamespace(
            is_group_leader=True,
            eager_host_mirror=True,
            resident_source_index=0,
            host_cache=torch.zeros(4 * block_size, width),
            request_state_indices=request_state_indices,
            begin_forward=MagicMock(),
            invalidate_written_slots=MagicMock(),
        )
        handle = HiSparseCacheHandle(runtime)
        handle.view = SimpleNamespace(
            block_size=block_size, cache=torch.zeros(4, block_size, width)
        )
        handle.slot_mapping = slots
        handle.mirror_slot_mapping = slots
        handle.draft_layer = draft_layer
        handles.append(handle)
    target, draft = handles
    # Rows the draft layer held before this step's drafter ran.
    draft.view.cache[resident_block] = -1.0

    worker = HiSparseConnectorWorker(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
            num_lookahead_tokens=3,
        ),
        MagicMock(),
    )
    worker.initialize(
        handles,
        ["target", "draft"],
        MagicMock(),
        max_num_reqs=1,
        host_num_blocks=4,
        device=torch.device("cpu"),
        pinned_host_pools=[],
    )
    connector = object.__new__(HiSparseConnector)
    connector.connector_worker = worker
    connector._get_connector_metadata = MagicMock(
        return_value=HiSparseConnectorMetadata(
            SparseKVOffloadCommand(
                [
                    SparseKVPageTransfer(
                        transfer_id=transfer_id,
                        host_block_id=host_block,
                        resident_block_ids=(resident_block,),
                        after_forward=True,
                    )
                ]
            ),
            (),
            (),
            {
                "req": (
                    SparseKVRowMirror(
                        source_starts=(resident_block * block_size,),
                        destination_start=host_block * block_size,
                        num_rows=block_size,
                    ),
                )
            },
            True,
            {},
        )
    )
    # A verification step: four query tokens of one request fill the page.
    attn_metadata = SimpleNamespace(
        num_actual_tokens=block_size,
        num_decode_tokens=0,
        num_reqs=1,
        max_query_len=block_size,
        req_id_per_token=torch.zeros(block_size, dtype=torch.int32),
    )

    connector.start_load_kv(
        SimpleNamespace(),
        request_state_indices=None,
        request_ids=["req"],
        num_tokens=block_size,
        attn_metadata={"target": attn_metadata, "draft": attn_metadata},
    )
    target.view.cache[resident_block] = 1.0
    if cg_mode != CUDAGraphMode.FULL:
        target.finish_kv_update()
    connector.finish_forward()
    enqueued_before_draft = list(worker._enqueued_transfer_ids)

    # The drafter writes the draft layer's rows for the same positions.
    if cg_mode != CUDAGraphMode.FULL:
        draft.prepare_group_for_batch(attn_metadata)
    draft.view.cache[resident_block] = 2.0
    if cg_mode != CUDAGraphMode.FULL:
        draft.finish_kv_update()
    # The next step's start mirrors the drafter's rows, then hands the page over.
    compute_stream.wait_event.reset_mock()
    connector._get_connector_metadata.return_value = HiSparseConnectorMetadata(
        None, (), (), {}, True, {}
    )
    connector.start_load_kv(
        SimpleNamespace(),
        request_state_indices=None,
        request_ids=[],
        num_tokens=0,
        attn_metadata={},
    )
    ((completion_event, _),) = worker._pending_transfer_events
    worker_meta = connector.build_connector_worker_meta()

    host_rows = slice(host_block * block_size, (host_block + 1) * block_size)
    torch.testing.assert_close(target.runtime.host_cache[host_rows], torch.ones(4, 8))
    torch.testing.assert_close(
        draft.runtime.host_cache[host_rows], torch.full((4, 8), 2.0)
    )
    draft_host = draft.runtime.host_cache
    draft_host_range = range(
        draft_host.data_ptr(),
        draft_host.data_ptr() + draft_host.numel() * draft_host.element_size(),
    )
    draft_copies = [
        index
        for index, (kind, ptr) in enumerate(log)
        if kind == "dma" and ptr in draft_host_range
    ]
    # One draft-layer copy, and the page is not handed back before it.
    assert len(draft_copies) == 1
    assert enqueued_before_draft == []
    assert log.index(("event", completion_event)) > draft_copies[0]
    # The next forward waits for the draft copy: the scheduler may already
    # have handed the blocks it reads and writes to another request.
    draft_copy_event = next(
        entry[1] for entry in log[draft_copies[0] :] if entry[0] == "event"
    )
    assert compute_stream.wait_event.call_args_list[0].args == (draft_copy_event,)
    assert worker_meta is not None
    assert worker_meta.enqueued_transfer_counts == {transfer_id: 1}
    assert worker_meta.completed_transfer_counts == {transfer_id: 1}


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("num_tokens", [0, 4])
def test_zero_token_step_holds_tp_ranks_after_host_write_wait(
    monkeypatch, shared, num_tokens
):
    """Steps without tokens hold every TP rank after its shared event wait."""
    calls: list[tuple[str, object]] = []
    compute_stream = MagicMock()
    compute_stream.wait_event.side_effect = lambda event: calls.append(("wait", event))
    tp_group = MagicMock()
    tp_group.barrier.side_effect = lambda: calls.append(("barrier", None))
    monkeypatch.setattr(worker_module, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(worker_module, "get_tp_group", lambda: tp_group)

    worker = object.__new__(HiSparseConnectorWorker)
    worker.shared_host_region = MagicMock() if shared else None
    worker.host_write_events = (MagicMock(), MagicMock())
    worker.host_write_event = worker.host_write_events[1]
    worker._next_host_write_event = 0
    worker._slot_mapping_staging = None
    worker.cache_handles = []
    worker._pending_invalid_block_ids = []
    for name in (
        "_stage_row_mirror_mapping",
        "_finish_previous_step",
        "_release_completed_dma_descriptors",
        "_set_row_mirrors",
        "_clear_forward_mirror_state",
        "_copy_host_blocks",
        "_restore_pages",
        "_submit_transfers",
    ):
        setattr(worker, name, MagicMock())

    worker.start_step(
        HiSparseConnectorMetadata(None, (), (), {}, True, {}),
        None,
        [],
        num_tokens=num_tokens,
    )

    expected = [("wait", worker.host_write_events[1])]
    if shared and not num_tokens:
        expected.append(("barrier", None))
    assert calls == expected


def test_scheduled_prefix_hit_publishes_adopted_copies():
    """Copies adopted after scheduling reach the worker as a residency update,
    leaving the block-table row the scheduler output carries untouched."""
    from tests.v1.core.test_prefix_caching import (
        HISPARSE_BLOCK_SIZE,
        _allocate_scheduled,
        _publish_hisparse_pages,
        make_hisparse_kv_cache_manager,
        make_request,
        sha256,
    )
    from vllm.v1.core.kv_cache_utils import init_none_hash

    init_none_hash(sha256)
    manager = make_hisparse_kv_cache_manager(32, 16, enable_caching=True)
    tokens = list(range(4 * HISPARSE_BLOCK_SIZE))
    original = make_request("original", tokens, HISPARSE_BLOCK_SIZE, sha256)
    assert _allocate_scheduled(manager, original, num_new_tokens=len(tokens))
    _publish_hisparse_pages(manager)
    copy_ids = [block.block_id for block in manager.get_blocks("original").blocks[2]]
    manager.free(original)

    resumed = make_request("resumed", tokens, HISPARSE_BLOCK_SIZE, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(resumed)
    num_new_tokens = len(tokens) - num_computed
    assert manager.allocate_slots(
        resumed,
        num_new_tokens=num_new_tokens,
        num_new_computed_tokens=num_computed,
        new_computed_blocks=computed,
    )
    scheduler = HiSparseConnectorScheduler(async_speculative=False)
    scheduler.bind_coordinator(get_hisparse_coordinator(manager))
    scheduler.requests[resumed.request_id] = resumed
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id=resumed.request_id,
                num_computed_tokens=num_computed,
                block_ids=manager.get_block_ids(resumed.request_id),
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], num_computed_tokens=[], new_block_ids=[]
        ),
        num_scheduled_tokens={resumed.request_id: num_new_tokens},
    )

    core_row = list(scheduler_output.scheduled_new_reqs[0].block_ids[2])

    metadata = scheduler.build_connector_meta(scheduler_output)

    update = metadata.residency_updates[resumed.request_id]
    assert update.pages == [0, 1, 2, 3]
    assert update.block_ids[0][:3] == copy_ids[:3]
    assert manager.get_block_ids(resumed.request_id)[2] == core_row
    assert core_row[:3] == [0, 0, 0]


def test_residency_updates_persist_by_state_row():
    """Updates name requests by batch row but are stored by state row.

    An update for a lost page and an appended page must leave the other pages
    and the other request's row intact across steps that reorder the batch.
    """
    table = torch.zeros((4, 2, 4), dtype=torch.int32)
    update_hisparse_residency(
        table,
        {
            "a": SparseKVResidencyUpdate([0, 1, 2], ([1, 2, 3], [11, 12, 13])),
            "b": SparseKVResidencyUpdate([0, 1], ([4, 5], [14, 15])),
        },
        ["a", "b"],
        torch.tensor([2, 0], dtype=torch.int32),
    )
    update_hisparse_residency(
        table,
        {"a": SparseKVResidencyUpdate([1, 3], ([0, 6], [0, 16]))},
        ["b", "a"],
        torch.tensor([0, 2], dtype=torch.int32),
    )

    assert table[[0, 2], 0].tolist() == [[4, 5, 0, 0], [1, 0, 3, 6]]
    assert table[[0, 2], 1].tolist() == [[14, 15, 0, 0], [11, 0, 13, 16]]


def _released_page_pool(async_load: bool, reuse: bool):
    """A reader whose clean pages HiSparse released (still readable), then a
    loader allocated either asynchronously (an external load) or for a forward,
    from those released blocks if ``reuse`` else from never-used ones."""
    from tests.v1.core.test_prefix_caching import (
        HISPARSE_BLOCK_SIZE,
        _allocate_scheduled,
        _publish_hisparse_pages,
        make_hisparse_kv_cache_manager,
        make_request,
        sha256,
    )
    from vllm.v1.core.kv_cache_utils import init_none_hash

    init_none_hash(sha256)
    manager = make_hisparse_kv_cache_manager(32, 16)
    scheduler = HiSparseConnectorScheduler(async_speculative=False)
    scheduler.bind_coordinator(get_hisparse_coordinator(manager))
    connector = object.__new__(HiSparseConnector)
    connector.connector_scheduler = scheduler

    tokens = list(range(6 * HISPARSE_BLOCK_SIZE))
    reader = make_request("reader", tokens, HISPARSE_BLOCK_SIZE, sha256)
    assert _allocate_scheduled(manager, reader, len(tokens))
    for hot_manager in scheduler.coordinator.hot_managers:
        hot_manager.require_hot(reader.request_id)
    assert _allocate_scheduled(manager, reader, len(tokens))
    scheduler.coordinator.take_residency_updates([reader.request_id])
    pool = manager.block_pool
    num_never_used = pool.get_num_free_blocks()
    _publish_hisparse_pages(manager)
    if reuse:
        pool.get_new_blocks(num_never_used)

    loader = make_request(
        "loader",
        list(range(1000, 1000 + HISPARSE_BLOCK_SIZE)),
        HISPARSE_BLOCK_SIZE,
        sha256,
    )
    if async_load:
        assert manager.allocate_slots(
            loader,
            0,
            num_external_computed_tokens=HISPARSE_BLOCK_SIZE,
            delay_cache_blocks=True,
        )
    else:
        assert manager.allocate_slots(loader, HISPARSE_BLOCK_SIZE)
    update = scheduler.coordinator.take_residency_updates([reader.request_id])
    lost = reader.request_id in update and 0 in update[reader.request_id].block_ids[0]
    assert lost == reuse
    connector.update_state_after_alloc(loader, manager.get_blocks(loader.request_id), 0)
    scheduled = {} if async_load else {loader.request_id: HISPARSE_BLOCK_SIZE}
    new_reqs = (
        []
        if async_load
        else [
            SimpleNamespace(
                req_id=loader.request_id,
                num_computed_tokens=0,
                block_ids=manager.get_block_ids(loader.request_id),
            )
        ]
    )
    scheduler_output = SimpleNamespace(
        kv_cache_block_copies=None,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=SimpleNamespace(
            new_block_ids=[], req_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens=scheduled,
    )
    return scheduler, scheduler_output


@pytest.mark.parametrize(
    "async_load,reuse,fenced",
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_async_load_into_released_page_waits_for_in_flight_step(
    async_load, reuse, fenced
):
    """A released page stays readable until the pool hands it out, so the step
    still on the GPU may read it. A forward that fills it runs after that step
    on the stream; an async external load (e.g. a NIXL read) is posted from the
    CPU, so its step must tell the worker to wait for the GPU first."""
    scheduler, scheduler_output = _released_page_pool(async_load, reuse)

    assert scheduler.build_connector_meta(scheduler_output).fence_async_loads == fenced

    idle_output = SimpleNamespace(
        kv_cache_block_copies=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            new_block_ids=[], req_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={},
    )
    assert not scheduler.build_connector_meta(idle_output).fence_async_loads


@pytest.mark.parametrize("fence", [False, True])
def test_fenced_step_waits_for_gpu_before_sibling_loads_start(monkeypatch, fence):
    """MultiConnector binds every connector before any starts its loads, so a
    wait at bind time orders a sibling's async load after the in-flight step."""
    events = []
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda *args: events.append("synchronize")
    )
    sibling = MagicMock()
    sibling.start_load_kv.side_effect = lambda *args, **kwargs: events.append("load")
    hisparse = object.__new__(HiSparseConnector)
    hisparse.start_load_kv = MagicMock()
    connector = object.__new__(MultiConnector)
    connector._connectors = [sibling, hisparse]
    connector._extra_async_saves = {}
    connector._connector_metadata = None
    metadata = HiSparseConnectorMetadata(
        None, (), (), {}, False, {}, fence_async_loads=fence
    )

    connector.bind_connector_metadata(
        MultiKVConnectorMetadata(metadata=(MagicMock(), metadata))
    )
    connector.start_load_kv(MagicMock())

    assert events == (["synchronize", "load"] if fence else ["load"])
