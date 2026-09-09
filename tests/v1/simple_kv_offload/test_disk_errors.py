# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O failures must reach the worker instead of stranding transfers."""

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
@pytest.mark.parametrize("failure", [None, "os_error", "short_io", "dma_error"])
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
    if failure == "dma_error":
        error = RuntimeError("DMA submission failed")
        disk_module.copy_blocks.side_effect = error
    submit(worker, is_store)

    if failure is not None:
        join_failed_thread(backend, is_store)
        other_thread = backend._load_thread if is_store else backend._store_thread
        assert other_thread.is_alive()
        operation = "store" if is_store else "load"
        # Poll repeatedly while the backend is running: errors must remain visible.
        for _ in range(2):
            with pytest.raises(
                RuntimeError, match=f"{operation} failed.*event 7"
            ) as exc:
                worker.get_finished(set())
            assert backend._disk_path in str(exc.value)
            if failure == "short_io":
                assert isinstance(exc.value.__cause__, OSError)
                assert "Short" in str(exc.value)
            else:
                assert exc.value.__cause__ is error
                assert str(error) in str(exc.value)
        assert not worker._load_events and not worker._store_events
        assert worker.build_connector_worker_meta() is None
        if failure != "dma_error":
            assert io_call.call_count == 2
        for direction in (False, True):
            with pytest.raises(RuntimeError, match=f"{operation} failed"):
                backend.launch_copy([1], [0], direction, 8, [])
    else:
        events = worker._store_events if is_store else worker._load_events
        assert events.ready.wait(timeout=5), "Transfer did not complete"
        assert backend._store_thread.is_alive() and backend._load_thread.is_alive()
        assert io_call.call_count == 2
        sending, receiving = worker.get_finished(set())
        assert sending is None
        assert receiving == (None if is_store else {"request-0"})
        meta = worker.build_connector_worker_meta()
        if is_store:
            assert meta is not None and meta.completed_store_events == {7: 1}
        else:
            assert meta is None
        assert worker.get_finished(set()) == (None, None)
        assert worker.build_connector_worker_meta() is None


@pytest.mark.parametrize(
    "first_store", [False, True], ids=["load-first", "store-first"]
)
def test_first_disk_failure_survives_other_thread_failure(
    monkeypatch, disk_worker, first_store
):
    worker, backend = disk_worker
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    errors = [OSError(errno.EIO, "read failed"), OSError(errno.ENOSPC, "write failed")]

    def fail(is_store):
        entered[is_store].set()
        assert release[is_store].wait(timeout=5)
        raise errors[is_store]

    for direction, syscall in ((False, "preadv"), (True, "pwritev")):
        monkeypatch.setattr(
            disk_module.os,
            syscall,
            lambda *args, direction=direction: fail(direction),
            raising=False,
        )
    try:
        completions: list[tuple[int, torch.Event]] = []
        for direction in (False, True):
            backend.launch_copy([0], [0], direction, 9, completions)
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
def test_disk_error_escapes_model_runner(monkeypatch, disk_worker, runner):
    """A later idle step must surface the failure even without a new transfer."""
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
        SimpleCPUOffloadConnector,
    )
    from vllm.v1.worker import kv_connector_model_runner_mixin as v1
    from vllm.v1.worker.gpu import kv_connector as v2

    worker, backend = disk_worker
    error = OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(
        disk_module.os, "pwritev", MagicMock(side_effect=error), raising=False
    )
    submit(worker, is_store=True)
    join_failed_thread(backend, is_store=True)

    # Bypass distributed initialization, keeping the real connector methods.
    connector = object.__new__(SimpleCPUOffloadConnector)
    connector.worker_handler = worker
    module = v1 if runner == "v1" else v2
    monkeypatch.setattr(module, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(module, "get_forward_context", lambda: None)
    scheduler_output = MagicMock(
        kv_connector_metadata=SimpleCPUOffloadMetadata(),
        finished_req_ids=set(),
        has_sync_kv_loads=False,
    )
    with pytest.raises(RuntimeError, match="store failed.*event 7") as exc:
        if runner == "v1":
            with v1.KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, wait_for_save=False
            ):
                pass
        else:
            monkeypatch.setattr(v2, "is_forward_context_available", lambda: True)
            active = v2.ActiveKVConnector(MagicMock(), {})
            if runner == "v2-no-forward":
                active.no_forward(scheduler_output)
            else:
                active.pre_forward(scheduler_output)
                active.post_forward(set())
    assert exc.value.__cause__ is error
    assert "No space left on device" in str(exc.value)
    assert worker.build_connector_worker_meta() is None
