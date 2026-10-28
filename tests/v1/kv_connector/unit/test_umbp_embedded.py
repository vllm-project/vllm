# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
import threading
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockTransferPlan,
    RankTopology,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import embedded
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.embedded import (
    _lookup_socket_path,
    _MoriLookupServer,
    _MoriSchedulerHandle,
    _MoriWorkerHandle,
)


@pytest.fixture
def blocked_native_load(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(embedded.time, "monotonic", lambda: clock.now)

    def get(keys, *args):
        started.set()
        assert release.wait(5)
        return [True] * len(keys)

    handle = _MoriWorkerHandle(
        SimpleNamespace(batch_get_ranges_into_ptr=get, close=lambda: None),
        "blocked-load",
        RankTopology(),
        str(tmp_path),
        1,
        1,
    )
    try:
        job = handle.load([BlockTransferPlan("running", 0)])
        future = handle._futures[id(job)]
        future.add_done_callback(lambda _: finished.set())
        assert started.wait(5)
        yield handle, job, future, release, finished, clock
    finally:
        release.set()
        handle.close()


def test_async_load_poll_timeout_keeps_buffers_owned(blocked_native_load):
    handle, job, future, release, finished, clock = blocked_native_load
    assert handle.poll(job) is None
    clock.now += 2
    with pytest.raises(TimeoutError, match="buffers still in use"):
        handle.poll(job)
    assert job.status is TransferJobStatus.RUNNING
    assert handle._futures[id(job)] is future
    assert not future.done()

    release.set()
    assert finished.wait(5)
    assert handle.poll(job) is job
    assert job.status is TransferJobStatus.COMPLETED
    assert id(job) not in handle._futures


def test_load_timeout_does_not_count_time_waiting_for_a_worker(blocked_native_load):
    handle, _, _, release, _, clock = blocked_native_load
    queued = handle.load([BlockTransferPlan("queued", 1)])
    finished = threading.Event()
    handle._futures[id(queued)].add_done_callback(lambda _: finished.set())
    clock.now += 2
    assert handle.poll(queued) is None
    assert queued.status is TransferJobStatus.RUNNING

    release.set()
    assert finished.wait(5)
    assert handle.poll(queued) is queued
    assert queued.status is TransferJobStatus.COMPLETED


@pytest.fixture
def socket_dir():
    """Keep AF_UNIX paths short regardless of CI's TMPDIR."""
    with tempfile.TemporaryDirectory(prefix="umbp-", dir="/tmp") as directory:
        yield directory


def test_lookup_server_refuses_a_socket_another_engine_serves(socket_dir):
    client = SimpleNamespace(batch_exists=lambda keys: [True] * len(keys))
    path = _lookup_socket_path("taken", (0, 0, 0, 0), socket_dir, "dp0")
    first = _MoriLookupServer(path, client)
    second = _MoriLookupServer(path, client)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="another live engine"):
            second.start()
        second.close()
        scheduler = _MoriSchedulerHandle("taken", RankTopology(), socket_dir, "dp0")
        assert scheduler.lookup(["k"]) == [True]
    finally:
        second.close()
        first.close()


def test_lookup_server_close_does_not_unlink_a_replacement(socket_dir):
    client = SimpleNamespace(batch_exists=lambda keys: [True] * len(keys))
    path = _lookup_socket_path("replaced", (0, 0, 0, 0), socket_dir)
    first = _MoriLookupServer(path, client)
    first.start()
    first.close()
    second = _MoriLookupServer(path, client)
    second.start()
    try:
        first.close()
        scheduler = _MoriSchedulerHandle("replaced", RankTopology(), socket_dir)
        assert scheduler.lookup(["k"]) == [True]
    finally:
        second.close()
