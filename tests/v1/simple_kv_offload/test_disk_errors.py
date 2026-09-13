# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk failures must drain DMA, fail transfers and allow subsequent work."""

import errno
import threading
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.simple_kv_offload import disk_backend as disk_module
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


class CompletionList(list):
    """Allow successful transfers to be awaited without stopping the backend."""

    def __init__(self):
        super().__init__()
        self.ready = threading.Event()

    def append(self, item):
        super().append(item)
        self.ready.set()


@pytest.fixture
def disk_worker(monkeypatch, tmp_path, request):
    # Keep the real threads, queues, staging pipeline and worker completion logic.
    def set_device(_):
        if getattr(request, "param", None) == "init_error":
            raise RuntimeError("device setup failed")

    monkeypatch.setattr(disk_module.current_platform, "set_device", set_device)
    monkeypatch.setattr(disk_module, "pin_tensor", MagicMock())
    monkeypatch.setattr(disk_module, "build_params", MagicMock())
    monkeypatch.setattr(disk_module, "copy_blocks", MagicMock())
    monkeypatch.setattr(torch, "Event", MagicMock)
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    backend = DiskBackend()
    worker = SimpleCPUOffloadWorker(MagicMock(), None, 4096, kv_offload_backend="disk")
    worker._backend = backend
    worker._load_events = CompletionList()
    worker._store_events = CompletionList()
    backend.init(
        gpu_caches={"k": torch.zeros(2, 4096, dtype=torch.int8)},
        device=torch.device("cpu"),
        load_stream=MagicMock(),
        store_stream=MagicMock(),
        disk_path=str(tmp_path / "cache.bin"),
        num_disk_slots=2,
        total_block_bytes=4096,
        num_buffer_slots=2,
        use_page_cache=True,
    )
    try:
        yield worker, backend
    finally:
        backend.shutdown()
        assert not backend._store_thread.is_alive()
        assert not backend._load_thread.is_alive()


def submit(worker, is_store):
    # Two blocks exercise an I/O failure after part of a transfer has succeeded.
    metadata = (
        SimpleCPUOffloadMetadata(
            store_event=7, store_gpu_blocks=[0, 1], store_cpu_blocks=[0, 1]
        )
        if is_store
        else SimpleCPUOffloadMetadata(
            load_event=7,
            load_gpu_blocks=[0, 1],
            load_cpu_blocks=[0, 1],
            load_event_to_reqs={7: ["request-0"]},
        )
    )
    worker.bind_connector_metadata(metadata)
    if is_store:
        worker.wait_for_save()
    else:
        worker.start_load_kv()


def join_failed_thread(backend, is_store):
    thread = backend._store_thread if is_store else backend._load_thread
    thread.join(timeout=5)
    assert not thread.is_alive(), "I/O thread did not exit after the injected failure"
    assert not backend._shutdown


@pytest.mark.parametrize("is_store", [False, True], ids=["load", "store"])
@pytest.mark.parametrize("failure", [None, "os_error", "short_io"])
def test_disk_io_result_reaches_worker(monkeypatch, disk_worker, is_store, failure):
    worker, backend = disk_worker
    error: Exception = OSError(
        errno.ENOSPC if is_store else errno.EIO, "disk I/O failed"
    )
    io_call = MagicMock(
        side_effect=[4096, error if failure == "os_error" else 0]
        if failure in ("os_error", "short_io")
        else None,
        return_value=4096,
    )
    monkeypatch.setattr(
        disk_module.os, "pwritev" if is_store else "preadv", io_call, raising=False
    )
    submit(worker, is_store)
    events = worker._store_events if is_store else worker._load_events
    assert events.ready.wait(timeout=5), "Transfer did not complete"
    assert backend._store_thread.is_alive() and backend._load_thread.is_alive()
    assert io_call.call_count == 2
    sending, receiving = worker.get_finished(set())
    assert sending is None
    assert receiving == (None if is_store else {"request-0"})
    assert worker.get_block_ids_with_load_errors() == (
        {1} if failure and not is_store else set()
    )
    meta = worker.build_connector_worker_meta()
    if is_store:
        assert meta is not None and meta.completed_store_events == {7: 1}
        assert meta.failed_store_events == ({7} if failure else set())
    else:
        assert meta is None
    assert worker.get_finished(set()) == (None, None)
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker.build_connector_worker_meta() is None

    # A later transfer uses the same buffers/threads and completes normally.
    io_call.side_effect = None
    events.ready.clear()
    metadata = SimpleCPUOffloadMetadata(
        store_event=8 if is_store else -1,
        load_event=-1 if is_store else 8,
        load_event_to_reqs={8: ["request-1"]},
    )
    worker.bind_connector_metadata(metadata)
    backend.launch_copy([0], [1], is_store, 8, events)
    assert events.ready.wait(timeout=5)
    assert worker.get_finished(set()) == (None, None if is_store else {"request-1"})
    assert worker.get_block_ids_with_load_errors() == set()
    meta = worker.build_connector_worker_meta()
    if is_store:
        assert meta.completed_store_events == {8: 1}
        assert meta.failed_store_events == set()


@pytest.mark.parametrize("is_store", [False, True], ids=["load", "store"])
def test_disk_failure_drains_dma_before_completion_and_buffer_reuse(
    monkeypatch, disk_worker, is_store
):
    worker, backend = disk_worker
    entered, release = threading.Event(), threading.Event()
    stream = backend._store_stream if is_store else backend._load_stream

    def drain():
        entered.set()
        assert release.wait(timeout=5)

    stream.synchronize.side_effect = drain
    io_call = MagicMock(side_effect=[4096, OSError(errno.EIO, "I/O error"), 4096])
    monkeypatch.setattr(
        disk_module.os, "pwritev" if is_store else "preadv", io_call, raising=False
    )
    submit(worker, is_store)
    events = worker._store_events if is_store else worker._load_events
    try:
        assert entered.wait(timeout=5)
        worker.bind_connector_metadata(
            SimpleCPUOffloadMetadata(
                store_event=8 if is_store else -1,
                load_event=-1 if is_store else 8,
                load_event_to_reqs={7: ["request-0"], 8: ["request-1"]},
            )
        )
        backend.launch_copy([0], [1], is_store, 8, events)
        assert worker.get_finished(set()) == (None, None)
        assert worker.get_block_ids_with_load_errors() == set()
        assert worker.build_connector_worker_meta() is None
        assert not events
        assert io_call.call_count == 2
        # Only a barrier in the other queue releases the blocked DMA drain.
        # Flushing just the already-published CUDA events would return early.
        other_stream = backend._load_stream if is_store else backend._store_stream
        other_stream.synchronize.side_effect = release.set
        worker.handle_preemptions(SimpleCPUOffloadMetadata(need_flush=True))
        assert release.is_set()
        assert io_call.call_count == 3
    finally:
        release.set()
    assert worker.get_finished(set()) == (
        None,
        None if is_store else {"request-0", "request-1"},
    )
    assert worker.get_block_ids_with_load_errors() == (set() if is_store else {1})
    meta = worker.build_connector_worker_meta()
    if is_store:
        assert meta.completed_store_events == {7: 1, 8: 1}
        assert meta.failed_store_events == {7}


@pytest.mark.parametrize(
    "first_store", [False, True], ids=["load-first", "store-first"]
)
def test_first_fatal_failure_survives_other_thread_failure(
    monkeypatch, disk_worker, first_store
):
    worker, backend = disk_worker
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    errors = [RuntimeError("load DMA failed"), RuntimeError("store DMA failed")]

    def fail(is_store):
        entered[is_store].set()
        assert release[is_store].wait(timeout=5)
        raise errors[is_store]

    for direction, stream in (
        (False, backend._load_stream),
        (True, backend._store_stream),
    ):
        stream.wait_event.side_effect = lambda *args, direction=direction: fail(
            direction
        )
    try:
        completions: list[tuple[int, torch.Event]] = []
        for direction in (False, True):
            backend.launch_copy([0], [0], direction, 9, completions, MagicMock())
        assert all(event.wait(timeout=5) for event in entered)
        assert worker.get_finished(set()) == (None, None)
        for direction in (False, True):
            backend.launch_copy([1], [1], direction, 10, completions)
        release[first_store].set()
        join_failed_thread(backend, first_store)
        with pytest.raises(RuntimeError) as first:
            worker.get_finished(set())
        assert first.value.__cause__ is errors[first_store]
        release[not first_store].set()
        join_failed_thread(backend, not first_store)
        with pytest.raises(RuntimeError) as second:
            worker.get_finished(set())
        assert second.value.__cause__ is errors[first_store]
        assert str(second.value) == str(first.value)
        assert not completions
    finally:
        for event in release:
            event.set()


@pytest.mark.parametrize("disk_worker", ["init_error"], indirect=True)
def test_disk_thread_initialization_failure_reaches_worker(disk_worker):
    worker, backend = disk_worker
    for direction in (False, True):
        join_failed_thread(backend, direction)
    with pytest.raises(RuntimeError, match="failed during initialization") as exc:
        worker.get_finished(set())
    assert str(exc.value.__cause__) == "device setup failed"
    assert backend._disk_path in str(exc.value)


@pytest.mark.parametrize("runner", ["v1", "v2", "v2-no-forward"])
@pytest.mark.parametrize("is_store", [False, True], ids=["load", "store"])
def test_disk_failure_reaches_model_runner_output(
    monkeypatch, disk_worker, runner, is_store
):
    """A later idle step must report failed transfers without raising."""
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
        SimpleCPUOffloadConnector,
    )
    from vllm.v1.worker import kv_connector_model_runner_mixin as v1
    from vllm.v1.worker.gpu import kv_connector as v2

    worker, backend = disk_worker
    monkeypatch.setattr(
        disk_module.os,
        "pwritev" if is_store else "preadv",
        MagicMock(side_effect=OSError(errno.EIO, "I/O error")),
        raising=False,
    )
    submit(worker, is_store)
    events = worker._store_events if is_store else worker._load_events
    assert events.ready.wait(timeout=5)

    # Bypass distributed initialization, keeping the real connector methods.
    connector = object.__new__(SimpleCPUOffloadConnector)
    connector.worker_handler = worker
    module = v1 if runner == "v1" else v2
    monkeypatch.setattr(module, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(module, "get_forward_context", lambda: None)
    scheduler_output = MagicMock(
        kv_connector_metadata=SimpleCPUOffloadMetadata(
            load_event_to_reqs={7: ["request-0"]}
        ),
        finished_req_ids=set(),
        has_sync_kv_loads=False,
    )
    if runner == "v1":
        with v1.KVConnectorModelRunnerMixin._get_kv_connector_output(
            scheduler_output, wait_for_save=False
        ) as output:
            pass
    else:
        monkeypatch.setattr(v2, "is_forward_context_available", lambda: True)
        active = v2.ActiveKVConnector(MagicMock(), {})
        if runner == "v2-no-forward":
            output = active.no_forward(scheduler_output).kv_connector_output
        else:
            active.pre_forward(scheduler_output)
            output = active.post_forward(set())
    assert output.finished_recving == (None if is_store else {"request-0"})
    assert output.invalid_block_ids == (set() if is_store else {0, 1})
    if is_store:
        assert output.kv_connector_worker_meta.completed_store_events == {7: 1}
        assert output.kv_connector_worker_meta.failed_store_events == {7}
    assert worker.build_connector_worker_meta() is None


@pytest.mark.parametrize("is_store", [False, True], ids=["load", "store"])
@pytest.mark.parametrize("failure", ["dma", "drain"])
def test_unsafe_device_failure_remains_fatal(
    monkeypatch, disk_worker, is_store, failure
):
    worker, backend = disk_worker
    error = RuntimeError("CUDA failure")
    if failure == "dma":
        disk_module.copy_blocks.side_effect = error
        monkeypatch.setattr(
            disk_module.os, "preadv", MagicMock(return_value=4096), raising=False
        )
    else:
        stream = backend._store_stream if is_store else backend._load_stream
        stream.synchronize.side_effect = error
        monkeypatch.setattr(
            disk_module.os,
            "pwritev" if is_store else "preadv",
            MagicMock(side_effect=OSError(errno.EIO, "I/O error")),
            raising=False,
        )
    submit(worker, is_store)
    join_failed_thread(backend, is_store)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="failed.*event 7") as exc:
            worker.get_finished(set())
        assert exc.value.__cause__ is error
        assert backend._disk_path in str(exc.value)
    assert not worker._load_events and not worker._store_events
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker.build_connector_worker_meta() is None
    with pytest.raises(RuntimeError):
        backend.synchronize()
    with pytest.raises(RuntimeError):
        backend.launch_copy([0], [0], is_store, 8, [])


@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize("policy", ["recompute", "fail"])
@pytest.mark.parametrize("failure", ["os_error", "short_io"])
def test_disk_read_failure_isolates_request_in_shared_batch(
    monkeypatch, disk_worker, num_groups, policy, failure
):
    """Healthy requests before and after a failed read keep their loaded KV."""
    from tests.v1.kv_connector.unit.utils import (
        create_model_runner_output,
        create_scheduler,
    )
    from tests.v1.simple_kv_offload.test_scheduler import (
        BLOCK_SIZE,
        _make_kv_cache_config,
        _make_vllm_config,
        make_request,
    )
    from vllm.v1.request import RequestStatus

    config = _make_vllm_config()
    config.kv_transfer_config.kv_load_failure_policy = policy
    config.kv_transfer_config.kv_connector_extra_config = {
        "cpu_bytes_to_use": 32768,
        "kv_offload_backend": "disk",
        "disk_path": "/unused-worker-path",
        "disk_capacity_bytes": 32768,
    }
    scheduler = create_scheduler(
        config, num_blocks=32, kv_cache_config=_make_kv_cache_config(32, num_groups)
    )
    manager = scheduler.connector.scheduler_manager
    requests = [make_request(num_blocks=2) for _ in range(3)]
    for request in requests:
        for group in range(num_groups):
            blocks = manager.cpu_block_pool.get_new_blocks(2)
            manager.cpu_block_pool.cache_full_blocks(
                request, blocks, 0, 2, BLOCK_SIZE, group
            )
            manager.cpu_block_pool.free_blocks(blocks)
        scheduler.add_request(request)
    scheduled = scheduler.schedule()
    metadata = scheduled.kv_connector_metadata
    assert metadata.load_event_to_reqs == {
        metadata.load_event: [request.request_id for request in requests]
    }
    for request in requests:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert request.num_computed_tokens == 2 * BLOCK_SIZE

    failed_request = requests[1]
    transfer = manager._reqs_to_load[failed_request.request_id].transfer_meta
    # Fail the last block of the middle request, after some of its DMA succeeds.
    failed_gpu_block = transfer.gpu_block_ids[-1]
    failed_disk_slot = transfer.cpu_block_ids[-1]
    cpu_blocks = {
        req_id: state.transfer_meta.cpu_block_ids
        for req_id, state in manager._reqs_to_load.items()
    }
    worker, backend = disk_worker

    def read_block(fd, views, offset):
        if offset == failed_disk_slot * backend._total_block_bytes:
            if failure == "os_error":
                raise OSError(errno.EIO, "read failed")
            return 0
        return backend._total_block_bytes

    io_call = MagicMock(side_effect=read_block)
    monkeypatch.setattr(disk_module.os, "preadv", io_call, raising=False)
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv()
    assert worker._load_events.ready.wait(timeout=5)
    _, received = worker.get_finished(set())
    assert received == {request.request_id for request in requests}
    failed = worker.get_block_ids_with_load_errors()
    assert failed == {failed_gpu_block}
    assert io_call.call_count == len(metadata.load_cpu_blocks)
    copied_gpu_blocks = [
        call.args[1][0] for call in disk_module.copy_blocks.call_args_list
    ]
    assert copied_gpu_blocks == [
        block for block in metadata.load_gpu_blocks if block != failed_gpu_block
    ]
    scheduler.update_from_output(
        scheduled,
        create_model_runner_output(
            [], finished_recving=received, invalid_block_ids=failed
        ),
    )
    assert manager.get_num_new_matched_tokens(failed_request, 0) == (0, False)
    assert not manager._reqs_to_load
    for request in requests:
        for bid in cpu_blocks[request.request_id]:
            block = manager.cpu_block_pool.blocks[bid]
            assert block.ref_cnt == 0
            assert (block.block_hash is None) == (request is failed_request)
    for healthy in (requests[0], requests[2]):
        assert healthy.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert healthy.num_computed_tokens == 2 * BLOCK_SIZE
    valid_tokens = BLOCK_SIZE if num_groups == 1 else 0
    if policy == "recompute":
        assert failed_request.num_computed_tokens == valid_tokens
    else:
        assert failed_request.status == RequestStatus.FINISHED_ERROR
    retried = scheduler.schedule()
    for healthy in (requests[0], requests[2]):
        assert healthy.status == RequestStatus.RUNNING
        assert retried.num_scheduled_tokens[healthy.request_id] == 1
    if policy == "recompute":
        assert failed_request.status == RequestStatus.RUNNING
        assert (
            retried.num_scheduled_tokens[failed_request.request_id]
            == failed_request.num_prompt_tokens - valid_tokens
        )
    else:
        assert failed_request.request_id not in retried.num_scheduled_tokens
    assert not retried.kv_connector_metadata.load_gpu_blocks
    assert not scheduler.failed_recving_kv_req_ids
