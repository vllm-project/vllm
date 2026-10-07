# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import ctypes
import mmap
import os
import uuid
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

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
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.hisparse import runtime as runtime_module
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.hisparse.types import (
    SparseKVOffloadCommand,
    SparseKVPageTransfer,
    SparseKVRowMirror,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
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
        return_value=HiSparseConnectorMetadata(None, (), (), {}, True)
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
    # This ordering test uses CPU tensors and the copy shim above. Registered
    # ROCm aliases and fused copies are exercised by the GPU worker tests.
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )
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
        None, (), (), {}, True
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


def test_scheduled_prefix_hit_publishes_adopted_copies():
    """Copies adopted after scheduling must reach the worker's block table."""
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

    scheduler.build_connector_meta(scheduler_output)

    resident_ids = scheduler_output.block_table_updates[resumed.request_id][2]
    assert resident_ids[:3] == copy_ids[:3]


@pytest.mark.parametrize("num_speculative_tokens", [None, 3])
@pytest.mark.parametrize("parallel_drafting", [False, True])
@pytest.mark.parametrize("batch_kind", ["decode", "mixed", "prefill"])
def test_hisparse_decode_metadata_skips_context_scan(
    monkeypatch, num_speculative_tokens, parallel_drafting, batch_kind
):
    """Decode keeps row mirrors while avoiding a context-length CPU scan."""
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
    manager = make_hisparse_kv_cache_manager(64, 32)
    coordinator = get_hisparse_coordinator(manager)
    config = SimpleNamespace(
        speculative_config=(
            None
            if num_speculative_tokens is None
            else SimpleNamespace(
                num_speculative_tokens=num_speculative_tokens,
                parallel_drafting=parallel_drafting,
            )
        ),
        use_v2_model_runner=True,
        num_lookahead_tokens=num_speculative_tokens or 0,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        kv_transfer_config=SimpleNamespace(),
    )
    connector = HiSparseConnector(
        config,
        KVConnectorRole.SCHEDULER,
        SimpleNamespace(hisparse_host_num_blocks=64),
    )
    scheduler = connector.connector_scheduler
    assert scheduler is not None
    max_decode_query_len = 1 + (num_speculative_tokens or 0)
    assert scheduler.max_decode_query_len == max_decode_query_len
    scheduler.bind_coordinator(coordinator)
    counts = [
        max_decode_query_len + int(batch_kind == "prefill"),
        max_decode_query_len + int(batch_kind != "decode"),
    ]
    starts = []
    for index, count in enumerate(counts):
        request = make_request(
            str(index),
            list(range(4 * HISPARSE_BLOCK_SIZE)),
            HISPARSE_BLOCK_SIZE,
            sha256,
        )
        assert _allocate_scheduled(manager, request, num_new_tokens=request.num_tokens)
        _publish_hisparse_pages(manager)
        scheduler.requests[request.request_id] = request
        starts.append(request.num_tokens - count)
    checked = []
    original = coordinator.all_context_pages_resident

    def record_scan(requests):
        checked.append(tuple(requests))
        return original(requests)

    monkeypatch.setattr(coordinator, "all_context_pages_resident", record_scan)
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["0", "1"], num_computed_tokens=starts, new_block_ids=[None, None]
        ),
        num_scheduled_tokens=dict(zip(["0", "1"], counts)),
    )
    metadata = scheduler.build_connector_meta(output)

    assert len(checked) == int(batch_kind == "prefill")
    assert metadata.all_context_pages_resident == (batch_kind == "prefill")
    assert output.has_sync_kv_loads
    assert set(metadata.row_mirrors) == {"0", "1"}
    for request_id, count in zip(["0", "1"], counts):
        assert sum(row.num_rows for row in metadata.row_mirrors[request_id]) == count


@pytest.mark.parametrize("retain_model_cache", [False, True])
def test_hisparse_shutdown_releases_shared_mmap(
    monkeypatch, retain_model_cache, caplog
):
    """DMA aliases must not retain a mapping after its last model view dies."""
    # No GPU work or registration is needed to test Python buffer ownership.
    # Use a real mmap and tensor views so an outstanding export prevents close.
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    region = SharedOffloadRegion(
        engine_id=str(uuid.uuid4()),
        num_chunks=2,
        rank=0,
        kv_bytes_per_chunk=2 * mmap.PAGESIZE,
        cpu_page_size=2 * mmap.PAGESIZE,
    )
    mapped = region.mmap_obj
    fd = region.fd
    worker = object.__new__(HiSparseConnectorWorker)
    worker._initialized = True
    worker._slot_mapping_staging = None
    worker.dma_stream = None
    worker._host_mirror_views = ()
    worker.shared_host_region = region
    worker.pinned_host_pools = []
    worker.cache_handles = [
        SimpleNamespace(
            runtime=SimpleNamespace(
                _host_cache=region.create_next_canonical_view(mmap.PAGESIZE),
                registered_host_pool=region.base_tensor,
                hot_backing=torch.empty(0),
            )
        )
        for _ in range(2)
    ]
    # _init_dma retains these aliases independently of the runtime.
    worker.host_caches = tuple(
        handle.runtime._host_cache for handle in worker.cache_handles
    )
    refs = [weakref.ref(cache) for cache in worker.host_caches]
    # Serving layers own distinct views until model-runner teardown, which
    # follows connector shutdown. Do not retain the mmap object in that case:
    # its destructor closes the mapping when the final exported view dies.
    model_caches = (
        [cache.view_as(cache) for cache in worker.host_caches]
        if retain_model_cache
        else []
    )
    mapped_ref = weakref.ref(mapped)
    if retain_model_cache:
        model_caches[0][0, 0] = 17
        mapped = None
    try:
        worker.shutdown()
        assert "Failed to close mmap_obj" not in caplog.text
        if retain_model_cache:
            assert mapped_ref() is not None
            assert model_caches[0][0, 0].item() == 17
            model_caches.clear()
            assert mapped_ref() is None, "Connector outlived the last model view"
        else:
            assert mapped.closed, "Connector retained an exported shared-pool view"
        assert all(ref() is None for ref in refs)
        with pytest.raises(OSError):
            os.fstat(fd)
        assert not worker._initialized
        # A second shutdown must not access already released runtime fields.
        worker.shutdown()
    finally:
        # Also close the real mapping when testing the unfixed implementation.
        worker.host_caches = ()
        model_caches.clear()
        for handle in worker.cache_handles:
            for name in ("_host_cache", "registered_host_pool", "hot_backing"):
                if hasattr(handle.runtime, name):
                    delattr(handle.runtime, name)
        region.cleanup()
        if mapped is not None and not mapped.closed:
            mapped.close()


@pytest.mark.parametrize("cleanup_path", ["release", "profiling", "initialization"])
def test_hisparse_non_shutdown_cleanup_keeps_strict_unmap_diagnostic(
    monkeypatch, caplog, cleanup_path
):
    from vllm.v1.hisparse.binding import release_hisparse_profiling_cache
    from vllm.v1.hisparse.runtime import release_pinned_state

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    region = SharedOffloadRegion(
        engine_id=str(uuid.uuid4()),
        num_chunks=1,
        rank=0,
        kv_bytes_per_chunk=mmap.PAGESIZE,
        cpu_page_size=mmap.PAGESIZE,
    )
    runtime = SimpleNamespace(
        _host_cache=region.base_tensor.view(-1),
        registered_host_pool=region.base_tensor,
        hot_backing=torch.empty(0),
        shared_host_region=region,
    )
    exported = runtime._host_cache.view_as(runtime._host_cache)
    mapped = weakref.ref(region.mmap_obj)
    try:
        if cleanup_path == "profiling":
            cache = SimpleNamespace(runtime=runtime)
            release_hisparse_profiling_cache(
                {"layer": SimpleNamespace(hisparse_cache=cache)}
            )
        elif cleanup_path == "initialization":
            cache = SimpleNamespace(runtime=runtime, view=torch.empty(1))
            worker = HiSparseConnectorWorker(
                SimpleNamespace(
                    compilation_config=SimpleNamespace(
                        static_forward_context={
                            "layer": SimpleNamespace(hisparse_cache=cache)
                        }
                    ),
                    scheduler_config=SimpleNamespace(max_num_seqs=1),
                ),
                SimpleNamespace(
                    hisparse_host_num_blocks=1,
                    kv_cache_groups=[
                        SimpleNamespace(
                            kv_cache_spec=worker_module.HiSparseHotSpec(
                                block_size=1, page_size=1, blocks_per_request=1
                            ),
                            layer_names=["layer" + worker_module.HISPARSE_HOT_SUFFIX],
                        )
                    ],
                ),
            )
            monkeypatch.setattr(
                worker_module, "get_tensor_model_parallel_rank", lambda: 0
            )

            def fail_initialize(*args, **kwargs):
                raise RuntimeError("initialization failed")

            monkeypatch.setattr(worker, "initialize", fail_initialize)
            with pytest.raises(RuntimeError, match="initialization failed"):
                worker.register_kv_caches({})
        else:
            release_pinned_state([runtime], [], region)
        assert "Failed to close mmap_obj" in caplog.text
        assert mapped() is not None
        del exported
        # Captured initialization tracebacks can retain caller-local views.
        # Final-view lifetime is covered separately without exception owners.
    finally:
        region.cleanup()
