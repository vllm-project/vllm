# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import enum
import gc
import sys
import threading
import time
from concurrent.futures import Future
from types import ModuleType, SimpleNamespace
from weakref import ref

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import (
    _kv_cache_config,
    _vllm_config,
    make_kv_caches,
    make_request,
    make_scheduler_output,
    run_in_new_process,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    TRANSFER_NOT_ATTEMPTED,
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVRange,
    KVRegion,
    RankTopology,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    StandaloneRuntime,
    UMBPRuntimeConfig,
    UMBPRuntimeFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.standalone import (
    _ABANDONED_STORAGES,
    _DaemonTransferPool,
    _SharedPoolWorkerHandle,
    normalize_endpoint,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
    _StoreBatch,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="UMBP requires ROCm"
)

ENDPOINT = "/run/umbp/test.grpc.sock"


class _DeploymentMode(enum.Enum):
    Local = 0
    StandaloneProcess = 1
    Distributed = 2


class _Server:
    """One host-wide pool shared by every fake client."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.up = True
        self.raises = False
        self.accept_puts = True
        self.clients: list[_Client] = []

    def lose(self, failure: str) -> None:
        if failure == "server-down":
            self.up = False
        elif failure == "client-raises":
            self.raises = True
        else:
            assert failure == "evicted"
            self.objects.clear()


class _Client:
    """Mirrors MORI, which fails batch calls to a lost server by result but
    can also raise, e.g. when an RPC fails outright."""

    def __init__(self, server: _Server, config) -> None:
        if not server.up:
            raise RuntimeError("standalone server unavailable")
        self.server = server
        self.config = config
        self.mode = _DeploymentMode.StandaloneProcess
        self.registered: dict[int, int] = {}
        self.flush_ok = True
        self.closed = False
        server.clients.append(self)

    def get_deployment_mode(self):
        return self.mode

    def _rpc(self) -> None:
        if self.server.raises:
            raise RuntimeError("standalone RPC failed")

    def _check_registered(self, ptr: int, length: int) -> None:
        assert any(
            base <= ptr and ptr + length <= base + size
            for base, size in self.registered.items()
        ), f"range 0x{ptr:x}+{length} is outside registered memory"

    def register_memory(self, ptr, size, location, device):
        self.registered[ptr] = size
        return True

    def deregister_memory(self, ptr):
        self._rpc()
        if not self.server.up:
            raise RuntimeError("standalone DeregisterMemory RPC failed")
        self.registered.pop(ptr, None)
        return True

    def batch_exists(self, keys):
        self._rpc()
        return [self.server.up and key in self.server.objects for key in keys]

    def batch_put_ranges_from_ptr(self, keys, object_sizes, pointers, sizes, offsets):
        self._rpc()
        if not (self.server.up and self.server.accept_puts):
            return [False] * len(keys)
        for key, size, ptrs, lengths, offs in zip(
            keys, object_sizes, pointers, sizes, offsets, strict=True
        ):
            payload = bytearray(size)
            for ptr, length, offset in zip(ptrs, lengths, offs, strict=True):
                self._check_registered(ptr, length)
                payload[offset : offset + length] = ctypes.string_at(ptr, length)
            self.server.objects.setdefault(key, bytes(payload))
        return [True] * len(keys)

    def batch_get_ranges_into_ptr(self, keys, pointers, sizes, offsets):
        self._rpc()
        results = []
        for key, ptrs, lengths, offs in zip(
            keys, pointers, sizes, offsets, strict=True
        ):
            payload = self.server.objects.get(key) if self.server.up else None
            if payload is None:
                results.append(False)
                continue
            for ptr, length, offset in zip(ptrs, lengths, offs, strict=True):
                self._check_registered(ptr, length)
                ctypes.memmove(ptr, payload[offset : offset + length], length)
            results.append(True)
        return results

    def flush(self):
        self._rpc()
        return self.server.up and self.flush_ok

    def clear(self):
        self._rpc()
        if not self.server.up:
            return False
        self.server.objects.clear()
        return True

    def close(self):
        self.closed = True


def _fake_mori(pool: _Server) -> SimpleNamespace:
    return SimpleNamespace(
        UMBPClient=lambda config: _Client(pool, config),
        UMBPConfig=lambda: SimpleNamespace(dram=SimpleNamespace()),
        UMBPStandaloneProcessConfig=lambda: SimpleNamespace(lookup_timeout_ms=1000),
        UMBPDeploymentMode=_DeploymentMode,
        MemoryLocationType=SimpleNamespace(CPU="cpu", GPU="gpu"),
    )


@pytest.fixture
def server(monkeypatch) -> _Server:
    pool = _Server()
    cpp = _fake_mori(pool)
    mori = sys.modules.get("mori") or ModuleType("mori")
    monkeypatch.setitem(sys.modules, "mori", mori)
    monkeypatch.setitem(sys.modules, "mori.cpp", cpp)
    monkeypatch.setattr(mori, "cpp", cpp, raising=False)
    return pool


def _runtime(**options) -> StandaloneRuntime:
    return StandaloneRuntime(
        UMBPRuntimeConfig("standalone", {"endpoint": ENDPOINT, **options})
    )


def _standalone_config(**options):
    return _vllm_config(
        {
            "mode": "standalone",
            "endpoint": ENDPOINT,
            "load_async": False,
            "key_namespace": "standalone-test",
            **options,
        }
    )


@pytest.fixture
def make_engine(server):
    """Build scheduler/worker connector pairs and shut them down afterwards."""
    connectors: list[UMBPStoreConnector] = []

    def make(**options):
        vllm_config = _standalone_config(**options)
        kv_config = _kv_cache_config()
        pair = (
            UMBPStoreConnector(vllm_config, KVConnectorRole.SCHEDULER, kv_config),
            UMBPStoreConnector(vllm_config, KVConnectorRole.WORKER, kv_config),
        )
        connectors.extend(pair)
        return pair

    yield make
    for connector in connectors:
        connector.shutdown()


def _store(scheduler, worker, request) -> UMBPConnectorWorkerMetadata:
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2],)),
        0,
    )
    worker.bind_connector_metadata(
        scheduler.build_connector_meta(
            make_scheduler_output(
                new_reqs=[request], scheduled_tokens={request.req_id: 32}
            )
        )
    )
    worker.wait_for_save()
    worker.connector_worker._drain_store_jobs(wait=True)
    metadata = worker.build_connector_worker_meta()
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=metadata, kv_cache_events=None)
    )
    return metadata


def _load(scheduler, worker, request) -> None:
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([5, 6],)),
        32,
    )
    worker.bind_connector_metadata(
        scheduler.build_connector_meta(
            make_scheduler_output(
                new_reqs=[request], scheduled_tokens={request.req_id: 32}
            )
        )
    )
    worker.start_load_kv(None)
    worker.wait_for_layer_load("layer0")


def _block_plan(key: str, source: torch.Tensor) -> BlockTransferPlan:
    return BlockTransferPlan(
        key,
        0,
        ranges=(KVRange("layer0", 0, 0, source.data_ptr(), 16, 16, 0),),
    )


@pytest.fixture
def stalled_shared_store(monkeypatch, tmp_path):
    """A one-thread shared-pool worker whose thread is stuck in a store."""
    started = threading.Event()
    release = threading.Event()
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    mori_calls: list[str] = []

    def put(keys, *args):
        started.set()
        assert release.wait(5)
        return [True] * len(keys)

    worker = _SharedPoolWorkerHandle(
        SimpleNamespace(
            batch_get_ranges_into_ptr=lambda keys, *args: [True] * len(keys),
            batch_put_ranges_from_ptr=put,
            flush=lambda: True,
            deregister_memory=lambda ptr: mori_calls.append("deregister_memory"),
            close=lambda: mori_calls.append("close"),
        ),
        "shared-stalled",
        RankTopology(),
        str(tmp_path),
        1,
        0.05,
    )
    worker._registered_storages.add(0x1000)
    stalled = worker.store([BlockTransferPlan("stalled", 0)])
    assert started.wait(5)
    try:
        yield worker, stalled, clock, mori_calls
    finally:
        release.set()
        worker.close()


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("settle", ["poll", "wait"])
def test_transfer_queued_behind_a_blocked_store_fails(
    stalled_shared_store, operation, settle
):
    """A job that never started holds no buffers, so it falls back at once."""
    worker, stalled, clock, _ = stalled_shared_store
    queued = getattr(worker, operation)([BlockTransferPlan("queued", 1)])
    assert worker.poll(queued) is None

    clock.now += 1
    result = worker.poll(queued) if settle == "poll" else worker.wait(queued)
    assert result is queued
    assert queued.status is TransferJobStatus.FAILED
    assert queued.failed_keys == {"queued"}
    assert queued.error.startswith(TRANSFER_NOT_ATTEMPTED)
    assert id(queued) not in worker._futures
    assert worker.poll(stalled) is None


def test_waiting_on_a_blocked_running_store_raises(stalled_shared_store):
    """MORI still reads the source blocks, so they must not be handed back."""
    worker, stalled, clock, _ = stalled_shared_store
    clock.now += 1

    with pytest.raises(TimeoutError):
        worker.wait(stalled)
    assert stalled.status is TransferJobStatus.RUNNING


def test_preempting_a_blocked_store_leaves_it_running(stalled_shared_store):
    """Preemption cancels a queued store but cannot stop one MORI is running."""
    worker, stalled, clock, _ = stalled_shared_store
    queued = worker.store([BlockTransferPlan("queued", 1)])
    clock.now += 1

    assert worker.cancel(stalled) is stalled
    assert stalled.status is TransferJobStatus.RUNNING
    assert worker.cancel(queued).status is TransferJobStatus.CANCELLED
    assert id(queued) not in worker._futures


def test_async_load_past_its_deadline_stays_pending(monkeypatch, tmp_path):
    """A stalled server must not fail the step; the request keeps its blocks."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)

    def get(keys, *args):
        started.set()
        assert release.wait(5)
        return [True] * len(keys)

    handle = _SharedPoolWorkerHandle(
        SimpleNamespace(
            batch_get_ranges_into_ptr=get,
            flush=lambda: True,
            close=lambda: None,
        ),
        "shared-load",
        RankTopology(),
        str(tmp_path),
        1,
        1,
    )
    worker = UMBPStoreConnectorWorker(handle)
    plan = BlockTransferPlan("restore", 1, request_id="req")
    try:
        worker.start_load_kv(
            None, UMBPConnectorMetadata(async_load=True, load_requests={"req": [plan]})
        )
        (job,) = worker._load_jobs["req"].values()
        handle._futures[id(job)].add_done_callback(lambda _: finished.set())
        assert started.wait(5)
        clock.now += 2

        assert worker.get_finished(set()) == (None, None)
        refused = handle.store([BlockTransferPlan("next", 2)])
        assert refused.status is TransferJobStatus.FAILED
        assert refused.error.startswith(TRANSFER_NOT_ATTEMPTED)

        release.set()
        assert finished.wait(5)
        assert worker.get_finished(set()) == (None, {"req"})
        assert worker.get_block_ids_with_load_errors() == set()
        assert not handle._load_deadlines
    finally:
        release.set()
        handle.close()


def test_worker_close_finishes_while_a_store_is_stuck(
    stalled_shared_store, monkeypatch
):
    """Shutdown must not raise, and must skip the MORI calls a stuck RPC blocks."""
    worker, stalled, clock, mori_calls = stalled_shared_store
    runtime_closed = threading.Event()
    close = worker.close

    def close_runtime():
        close()
        runtime_closed.set()

    monkeypatch.setattr(worker, "close", close_runtime)
    connector_worker = UMBPStoreConnectorWorker(worker)
    connector_worker._pending_stores.append(_StoreBatch(-1, {"request": stalled}))
    clock.now += 1

    connector_worker.close()

    assert runtime_closed.is_set()
    assert mori_calls == []


def test_close_releases_the_client_after_cancelling_queued_transfers(
    monkeypatch, tmp_path
):
    """A transfer cancelled before it started must not count as blocked."""
    started = threading.Event()
    release = threading.Event()
    mori_calls: list[str] = []

    def put(keys, *args):
        started.set()
        assert release.wait(5)
        return [True] * len(keys)

    worker = _SharedPoolWorkerHandle(
        SimpleNamespace(
            batch_put_ranges_from_ptr=put,
            flush=lambda: True,
            deregister_memory=lambda ptr: mori_calls.append("deregister_memory"),
            close=lambda: mori_calls.append("close"),
        ),
        "shared-close",
        RankTopology(),
        str(tmp_path),
        1,
        5,
    )
    storage = torch.zeros(16, dtype=torch.uint8).untyped_storage()
    worker._registered_storages.add(storage.data_ptr())
    worker._storages[storage.data_ptr()] = storage
    running = worker.store([BlockTransferPlan("running", 0)])
    assert started.wait(5)
    queued = worker.store([BlockTransferPlan("queued", 1)])
    queued_future = worker._futures[id(queued)]
    shutdown = worker._executor.shutdown

    def shutdown_then_finish(*args, **kwargs):
        shutdown(*args, **kwargs)
        release.set()

    monkeypatch.setattr(worker._executor, "shutdown", shutdown_then_finish)
    abandoned = len(_ABANDONED_STORAGES)

    worker.close()

    assert queued_future.cancelled()
    assert running.status is TransferJobStatus.COMPLETED
    assert mori_calls == ["deregister_memory", "close"]
    assert len(_ABANDONED_STORAGES) == abandoned


@pytest.mark.parametrize("stuck", ["store", "deregister_memory"])
def test_close_pins_kv_buffers_while_the_server_is_stalled(server, monkeypatch, stuck):
    """The server may still write into the KV buffers once it resumes."""
    worker = _runtime(timeout_ms=20).create_worker_handle("ns", RankTopology(), None)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    release = threading.Event()
    try:
        if stuck == "store":
            started = _stalled_store(server, monkeypatch, release)
            worker.store([_block_plan("slow", source)])
            assert started.wait(5)
        else:
            _stall_scheduler_call(server, monkeypatch, stuck, release)

        worker.close()

        storage_ptr = source.untyped_storage().data_ptr()
        assert any(s.data_ptr() == storage_ptr for s in _ABANDONED_STORAGES)
        assert storage_ptr in server.clients[0].registered
    finally:
        release.set()


@pytest.mark.parametrize("role", ["scheduler", "worker"])
def test_close_releases_the_client_kv_buffers_and_threads(server, role):
    """MORI drops a connection only once its client object is destroyed."""
    runtime = _runtime()
    if role == "scheduler":
        handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
        threads = handle._calls._threads
        storage = None
    else:
        handle = runtime.create_worker_handle("ns", RankTopology(), None)
        threads = handle._executor._threads
        source = torch.zeros(16, dtype=torch.uint8)
        handle.register_buffers({"layer0": source})
        storage = ref(source.untyped_storage())
        del source
    client = ref(server.clients.pop())

    handle.close()

    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    gc.collect()
    assert client() is None
    assert storage is None or storage() is None


@pytest.mark.parametrize(
    ("role", "call"), [("scheduler", "close"), ("worker", "deregister_memory")]
)
def test_close_waits_for_a_slow_but_live_server(server, monkeypatch, role, call):
    runtime = _runtime()
    if role == "scheduler":
        handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
    else:
        handle = runtime.create_worker_handle("ns", RankTopology(), None)
        handle.register_buffers({"layer0": torch.zeros(16, dtype=torch.uint8)})
    client = server.clients[0]
    original = getattr(client, call)
    called = threading.Event()
    answered = threading.Event()

    def slow(*args):
        called.set()
        assert answered.wait(5)
        return original(*args)

    monkeypatch.setattr(client, call, slow)

    def answer() -> None:
        if called.wait(5):
            answered.set()

    server_thread = threading.Thread(target=answer)
    server_thread.start()
    abandoned = len(_ABANDONED_STORAGES)

    handle.close()

    assert client.closed
    assert client.registered == {}
    assert len(_ABANDONED_STORAGES) == abandoned
    server_thread.join()


def test_preempted_store_never_reaches_the_server(server, monkeypatch):
    worker = _runtime(num_workers=1).create_worker_handle("ns", RankTopology(), None)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    release = threading.Event()
    started = _stalled_store(server, monkeypatch, release)
    try:
        running = worker.store([_block_plan("running", source)])
        assert started.wait(5)
        preempted = worker.store([_block_plan("preempted", source)])
        assert worker.cancel(preempted).status is TransferJobStatus.CANCELLED
        release.set()

        after = worker.wait(worker.store([_block_plan("after", source)]))
        assert after.status is TransferJobStatus.COMPLETED
        assert worker.wait(running).status is TransferJobStatus.COMPLETED
        assert server.objects.keys() == {"running", "after"}
    finally:
        release.set()
        worker.close()


def _hang(*args) -> None:
    threading.Event().wait()


def _close_against_a_hung_server(role: str, stuck: str) -> None:
    pool = _Server()
    sys.modules["mori"] = ModuleType("mori")
    sys.modules["mori.cpp"] = _fake_mori(pool)
    runtime = _runtime(lookup_timeout_ms=100, timeout_ms=100)
    if role == "scheduler":
        handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
        setattr(pool.clients[0], stuck, _hang)
        assert handle.lookup(["k"]) == [False]
    else:
        handle = runtime.create_worker_handle("ns", RankTopology(), None)
        source = torch.arange(16, dtype=torch.uint8)
        handle.register_buffers({"layer0": source})
        setattr(pool.clients[0], stuck, _hang)
        if stuck == "batch_put_ranges_from_ptr":
            job = handle.store([_block_plan("stuck", source)])
            assert handle.poll(job) is None
    handle.close()


@pytest.mark.parametrize(
    ("role", "stuck"),
    [
        ("worker", "batch_put_ranges_from_ptr"),
        ("worker", "deregister_memory"),
        ("scheduler", "batch_exists"),
        ("scheduler", "close"),
    ],
)
def test_process_exits_while_the_server_never_returns(role, stuck):
    """Neither close() nor interpreter exit may wait on a call stuck in MORI."""
    run_in_new_process(_close_against_a_hung_server, role, stuck, timeout=60)


@pytest.mark.parametrize("max_workers", [0, -1])
def test_transfer_pool_rejects_non_positive_worker_counts(max_workers):
    with pytest.raises(ValueError, match="max_workers"):
        _DaemonTransferPool(max_workers, "umbp-test")


def test_blocked_transfer_count_is_safe_during_job_updates(tmp_path):
    """Transfer threads add jobs while the scheduler counts blocked ones."""
    worker = _SharedPoolWorkerHandle(
        SimpleNamespace(close=lambda: None),
        "shared-count",
        RankTopology(),
        str(tmp_path),
        1,
        1,
    )
    finished: Future = Future()
    finished.set_result(None)
    worker._store_deadlines[-1] = 0
    worker._futures[-1] = finished
    added = threading.Event()

    def add_jobs():
        for job_id in range(5000):
            with worker._job_lock:
                worker._futures[job_id] = Future()
                worker._store_deadlines[job_id] = 0
        added.set()

    adder = threading.Thread(target=add_jobs)
    adder.start()
    try:
        while not added.is_set():
            worker._blocked_transfers()
    finally:
        adder.join()

    assert worker._blocked_transfers() == 5000
    worker.close()


@pytest.mark.parametrize(
    ("endpoint", "address"),
    [
        ("/run/umbp/a.grpc.sock", "unix:///run/umbp/a.grpc.sock"),
        ("unix:///run/umbp/a.grpc.sock", "unix:///run/umbp/a.grpc.sock"),
    ],
)
def test_endpoint_normalizes_to_a_unix_address(endpoint, address):
    assert normalize_endpoint(endpoint) == address


@pytest.mark.parametrize(
    "endpoint",
    ["tcp:///run/umbp/a.sock", "run/umbp/a.sock", "unix://run/umbp/a.sock"],
)
def test_endpoint_rejects_non_socket_paths(endpoint):
    with pytest.raises(ValueError, match="endpoint"):
        normalize_endpoint(endpoint)


def _build(vllm_config):
    return UMBPRuntimeFactory.build(UMBPRuntimeConfig.from_vllm(vllm_config))


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"endpoint": None}, "requires endpoint"),
        ({"endpoint": 5}, "requires endpoint"),
        ({"capacity_bytes": 1024}, "owns its DRAM pool"),
        ({"total_capacity_bytes": 1024}, "owns its DRAM pool"),
        ({"dram_prefault": False}, "owns its DRAM pool"),
        ({"auto_start": True}, "owns its DRAM pool"),
        ({"startup_timeout_ms": 0}, "startup_timeout_ms"),
        ({"lookup_timeout_ms": 0}, "lookup_timeout_ms"),
        ({"lookup_timeout_ms": -1}, "lookup_timeout_ms"),
        ({"lookup_timeout_ms": True}, "lookup_timeout_ms"),
        ({"timeout_ms": 0}, "timeout_ms"),
        ({"num_workers": 0}, "num_workers"),
        ({"num_workers": -1}, "num_workers"),
        ({"master_address": "x"}, "distributed-only"),
    ],
)
def test_config_rejects_invalid_standalone_options(options, match):
    with pytest.raises(ValueError, match=match):
        _build(_standalone_config(**options))


@pytest.mark.parametrize("lookup_timeout_ms", [None, 1])
def test_client_targets_the_server_without_sizing_a_local_pool(
    server, lookup_timeout_ms
):
    options = (
        {} if lookup_timeout_ms is None else {"lookup_timeout_ms": lookup_timeout_ms}
    )
    runtime = _runtime(startup_timeout_ms=500, **options)
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)

    config = server.clients[0].config
    assert config.standalone_process.address == "unix://" + ENDPOINT
    assert config.standalone_process.auto_start is False
    assert config.standalone_process.startup_timeout_ms == 500
    assert config.standalone_process.lookup_timeout_ms == (
        1000 if lookup_timeout_ms is None else lookup_timeout_ms
    )
    assert vars(config.dram) == {}
    handle.close()
    assert server.clients[0].closed


def test_standalone_attaches_with_mori_lacking_lookup_deadlines(server, monkeypatch):
    monkeypatch.setattr(
        sys.modules["mori.cpp"], "UMBPStandaloneProcessConfig", SimpleNamespace
    )
    handle = _runtime(lookup_timeout_ms=25).create_scheduler_handle(
        "ns", RankTopology(), None
    )

    config = server.clients[0].config
    assert config.standalone_process.address == "unix://" + ENDPOINT
    assert not hasattr(config.standalone_process, "lookup_timeout_ms")
    handle.close()


def test_unreachable_server_fails_at_startup(server):
    server.up = False
    runtime = _runtime()
    with pytest.raises(RuntimeError, match="cannot attach"):
        runtime.create_scheduler_handle("ns", RankTopology(), None)


def test_mori_without_umbp_fails_at_startup(server, monkeypatch):
    monkeypatch.setitem(sys.modules, "mori.cpp", None)
    with pytest.raises(RuntimeError, match="BUILD_UMBP"):
        _runtime().create_scheduler_handle("ns", RankTopology(), None)


def test_client_in_the_wrong_deployment_mode_is_rejected(server, monkeypatch):
    original = sys.modules["mori.cpp"].UMBPClient

    def local_client(config):
        client = original(config)
        client.mode = _DeploymentMode.Local
        return client

    monkeypatch.setattr(sys.modules["mori.cpp"], "UMBPClient", local_client)
    runtime = _runtime()
    with pytest.raises(RuntimeError, match="expected StandaloneProcess"):
        runtime.create_worker_handle("ns", RankTopology(), None)


def test_worker_registers_whole_kv_storages_with_the_server(server):
    worker = _runtime().create_worker_handle("ns", RankTopology(), None)
    backing = torch.zeros(128, dtype=torch.uint8)
    storage = backing.untyped_storage()

    worker.register_buffers({"layer0": backing[32:96]})

    assert server.clients[0].registered == {storage.data_ptr(): storage.nbytes()}
    worker.close()
    assert server.clients[0].registered == {}


def test_rejected_kv_registration_fails_at_startup(server, monkeypatch):
    """The server could not reach unregistered KV, so every transfer would fail."""
    worker = _runtime().create_worker_handle("ns", RankTopology(), None)
    monkeypatch.setattr(server.clients[0], "register_memory", lambda *args: False)

    with pytest.raises(RuntimeError, match="failed to register"):
        worker.register_buffers({"layer0": torch.zeros(16, dtype=torch.uint8)})
    worker.close()


def test_store_waits_for_kv_written_on_the_compute_stream(server, monkeypatch):
    """The server must not copy GPU KV the forward pass has not finished writing."""
    worker = _runtime().create_worker_handle("ns", RankTopology(), None)
    source = torch.zeros(16, dtype=torch.uint8, device="cuda:0")
    worker.register_buffers({"layer0": source})
    stored: list[torch.Tensor] = []

    def put(keys, *args):
        stored.append(source.cpu())
        return [True] * len(keys)

    monkeypatch.setattr(server.clients[0], "batch_put_ranges_from_ptr", put)
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            torch.cuda._sleep(1 << 28)
            source.fill_(7)
            job = worker.wait(worker.store([_block_plan("k", source)]))

        assert job.status is TransferJobStatus.COMPLETED
        assert stored[0].tolist() == [7] * 16
    finally:
        worker.close()


def test_close_deregisters_every_storage_when_one_fails(server, monkeypatch):
    """One failed deregistration must not leave the other KV pinned."""
    worker = _runtime().create_worker_handle("ns", RankTopology(), None)
    first, second = (
        torch.zeros(64, dtype=torch.uint8),
        torch.zeros(64, dtype=torch.uint8),
    )
    worker.register_buffers({"layer0": first, "layer1": second})
    client = server.clients[0]
    deregister = client.deregister_memory
    failed: list[int] = []

    def fail_first(ptr):
        if not failed:
            failed.append(ptr)
            raise RuntimeError("standalone DeregisterMemory RPC failed")
        return deregister(ptr)

    monkeypatch.setattr(client, "deregister_memory", fail_first)

    worker.close()

    assert client.registered.keys() == set(failed)


def test_publish_rejects_an_incomplete_job(server):
    worker = _runtime().create_worker_handle("ns", RankTopology(), None)
    job = TransferJobState((BlockTransferPlan("k", 0),))

    with pytest.raises(RuntimeError, match="incomplete"):
        worker.publish(job)
    worker.close()


def test_failed_clear_reports_false(server, monkeypatch):
    runtime = _runtime()
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
    monkeypatch.setattr(server.clients[0], "clear", lambda: False)

    assert handle.clear() is False


def test_engines_share_objects_through_one_server(make_engine):
    producer_scheduler, producer_worker = make_engine()
    consumer_scheduler, consumer_worker = make_engine()
    other_model_scheduler, _ = make_engine(key_namespace="another-model")
    source = make_kv_caches(fill=True)
    destination = make_kv_caches()
    producer_worker.register_kv_caches(source)
    consumer_worker.register_kv_caches(destination)
    hashes = [b"shared-a", b"shared-b"]

    _store(
        producer_scheduler,
        producer_worker,
        make_request("producer", num_tokens=33, block_hashes=hashes),
    )
    other = make_request("other", num_tokens=33, block_hashes=hashes)
    assert other_model_scheduler.get_num_new_matched_tokens(other, 0) == (0, False)
    consumer = make_request("consumer", num_tokens=33, block_hashes=hashes)
    assert consumer_scheduler.get_num_new_matched_tokens(consumer, 0) == (32, False)
    _load(consumer_scheduler, consumer_worker, consumer)

    for name in source:
        assert torch.equal(source[name][1], destination[name][5])
        assert torch.equal(source[name][2], destination[name][6])
    assert consumer_worker.get_block_ids_with_load_errors() == set()
    assert consumer_worker.get_kv_connector_kv_cache_events() is None


def test_reset_cache_clears_the_pool_for_every_engine(make_engine):
    scheduler_a, worker_a = make_engine()
    scheduler_b, _ = make_engine()
    worker_a.register_kv_caches(make_kv_caches(fill=True))
    hashes = [b"reset-a", b"reset-b"]
    _store(
        scheduler_a,
        worker_a,
        make_request("producer", num_tokens=33, block_hashes=hashes),
    )
    first = make_request("r1", num_tokens=33, block_hashes=hashes)
    assert scheduler_b.get_num_new_matched_tokens(first, 0) == (32, False)

    assert scheduler_a.reset_cache() is True

    second = make_request("r2", num_tokens=33, block_hashes=hashes)
    assert scheduler_b.get_num_new_matched_tokens(second, 0) == (0, False)


def _stall_scheduler_call(server, monkeypatch, method, release) -> list:
    """Record the calls that reach the server, and "returned" once one does."""
    client = server.clients[0]
    original = getattr(client, method)
    calls: list = []

    def stalled(*args):
        calls.append(args)
        try:
            assert release.wait(10)
            return original(*args)
        finally:
            calls.append("returned")

    monkeypatch.setattr(client, method, stalled)
    return calls


def test_lookup_without_a_reply_misses_until_the_server_answers(server, monkeypatch):
    handle = _runtime(lookup_timeout_ms=50).create_scheduler_handle(
        "ns", RankTopology(), None
    )
    server.objects["k"] = b"x"
    release = threading.Event()
    calls = _stall_scheduler_call(server, monkeypatch, "batch_exists", release)
    try:
        assert handle.lookup(["k"]) == [False]
        assert handle.lookup(["j"]) == [False]
        assert "returned" not in calls

        release.set()
        deadline = time.monotonic() + 5
        while handle.lookup(["k"]) != [True]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # The second lookup failed at once rather than queueing behind the first.
        assert (["j"],) not in calls
    finally:
        release.set()
        handle.close()


def test_clear_without_a_reply_reports_false(server, monkeypatch):
    handle = _runtime(lookup_timeout_ms=50).create_scheduler_handle(
        "ns", RankTopology(), None
    )
    release = threading.Event()
    calls = _stall_scheduler_call(server, monkeypatch, "clear", release)
    try:
        assert handle.clear() is False
        assert "returned" not in calls
    finally:
        release.set()
        handle.close()


@pytest.mark.parametrize("stuck", ["batch_exists", "close"])
def test_scheduler_close_does_not_wait_on_a_stalled_server(server, monkeypatch, stuck):
    handle = _runtime(lookup_timeout_ms=50, timeout_ms=50).create_scheduler_handle(
        "ns", RankTopology(), None
    )
    release = threading.Event()
    calls = _stall_scheduler_call(server, monkeypatch, stuck, release)
    try:
        if stuck == "batch_exists":
            assert handle.lookup(["k"]) == [False]

        handle.close()

        assert "returned" not in calls
        assert not server.clients[0].closed
    finally:
        release.set()


def test_scheduler_drops_its_client_once_a_stalled_call_returns(server, monkeypatch):
    handle = _runtime(lookup_timeout_ms=50).create_scheduler_handle(
        "ns", RankTopology(), None
    )
    release = threading.Event()
    _stall_scheduler_call(server, monkeypatch, "batch_exists", release)
    assert handle.lookup(["k"]) == [False]

    handle.close()
    release.set()

    for thread in handle._calls._threads:
        thread.join(5)
        assert not thread.is_alive()
    monkeypatch.undo()
    client = ref(server.clients.pop())
    gc.collect()
    assert client() is None


@pytest.mark.parametrize("failure", ["server-down", "client-raises"])
def test_lookup_and_clear_degrade_when_the_server_is_lost(server, failure):
    handle = _runtime().create_scheduler_handle("ns", RankTopology(), None)
    server.objects["k"] = b"x"
    assert handle.lookup(["k"]) == [True]

    server.lose(failure)

    assert handle.lookup(["k", "j"]) == [False, False]
    assert handle.clear() is False
    handle.close()


def test_lookup_rejects_a_short_result(server, monkeypatch):
    runtime = _runtime()
    handle = runtime.create_scheduler_handle("ns", RankTopology(), None)
    monkeypatch.setattr(server.clients[0], "batch_exists", lambda keys: [True])

    assert handle.lookup(["a", "b"]) == [False, False]


@pytest.mark.parametrize("load_async", [False, True])
@pytest.mark.parametrize("failure", ["server-down", "client-raises", "evicted"])
def test_lost_objects_after_a_hit_mark_blocks_for_recompute(
    server, make_engine, failure, load_async
):
    """Objects gone between lookup and load must not count as restored KV."""
    producer_scheduler, producer_worker = make_engine()
    consumer_scheduler, consumer_worker = make_engine(load_async=load_async)
    producer_worker.register_kv_caches(make_kv_caches(fill=True))
    consumer_worker.register_kv_caches(make_kv_caches())
    hashes = [b"lost-a", b"lost-b"]
    _store(
        producer_scheduler,
        producer_worker,
        make_request("producer", num_tokens=33, block_hashes=hashes),
    )
    consumer = make_request("consumer", num_tokens=33, block_hashes=hashes)
    assert consumer_scheduler.get_num_new_matched_tokens(consumer, 0) == (
        32,
        load_async,
    )
    server.lose(failure)

    _load(consumer_scheduler, consumer_worker, consumer)
    if load_async:
        deadline = time.monotonic() + 5
        while consumer_worker.get_finished(set()) != (None, {"consumer"}):
            assert time.monotonic() < deadline
            time.sleep(0.01)

    assert consumer_worker.get_block_ids_with_load_errors() == {5, 6}


@pytest.mark.parametrize("failure", ["server-down", "client-raises", "puts-rejected"])
def test_failed_store_is_reported_without_load_errors(server, make_engine, failure):
    # vLLM fails or recomputes requests owning blocks reported as load errors.
    # A failed store leaves the source KV intact, so it must not report any.
    scheduler, worker = make_engine()
    worker.register_kv_caches(make_kv_caches(fill=True))
    if failure == "puts-rejected":
        server.accept_puts = False
    else:
        server.lose(failure)

    metadata = _store(
        scheduler,
        worker,
        make_request("producer", num_tokens=33, block_hashes=[b"dead-a", b"dead-b"]),
    )

    assert worker.get_block_ids_with_load_errors() == set()
    (result,) = metadata.store_events.values()
    assert result.completed_workers == 1
    assert result.failed_objects == {(0, b"dead-a"), (0, b"dead-b")}
    assert server.objects == {}


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("flush", "MORI UMBP flush failed"),
        ("puts-rejected", "MORI UMBP range store failed"),
    ],
)
def test_store_failure_is_reported(server, failure, error):
    runtime = _runtime()
    topology = RankTopology()
    layout = KVLayoutDescriptor(
        regions=(KVRegion("layer0", 0, 16, 16, 0),), topology=topology
    )
    worker = runtime.create_worker_handle("ns", topology, layout)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    plan = _block_plan("publish-key", source)
    try:
        if failure == "flush":
            server.clients[0].flush_ok = False
        else:
            server.accept_puts = False
        job = worker.wait(worker.store([plan]))

        assert job.status is TransferJobStatus.FAILED
        assert job.failed_keys == {plan.key}
        assert job.completed_keys == set()
        assert job.completed_bytes == 0
        assert job.error == error
    finally:
        worker.close()


@pytest.mark.parametrize("frees_pool", ["backoff", "flush"])
def test_store_retries_puts_rejected_until_the_pool_has_room(
    server, monkeypatch, frees_pool
):
    worker = _runtime().create_worker_handle("ns", RankTopology(), None)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    server.accept_puts = False

    def free_pool(*_):
        server.accept_puts = True
        return True

    if frees_pool == "backoff":
        monkeypatch.setattr(time, "sleep", free_pool)
    else:
        monkeypatch.setattr(server.clients[0], "flush", free_pool)
    try:
        job = worker.wait(worker.store([_block_plan("full", source)]))

        assert job.status is TransferJobStatus.COMPLETED
        assert server.objects == {"full": bytes(range(16))}
    finally:
        worker.close()


def test_store_blocks_without_a_block_layout_requests_ranges(server):
    worker = _runtime().create_worker_handle("ns", RankTopology(), None)
    try:
        assert worker.store_blocks([BlockTransferPlan("k", 0, group_id=0)]) is None
    finally:
        worker.close()


def _stalled_store(server, monkeypatch, release: threading.Event) -> threading.Event:
    client = server.clients[0]
    put = client.batch_put_ranges_from_ptr
    started = threading.Event()

    def blocking_put(*args):
        started.set()
        assert release.wait(10)
        return put(*args)

    monkeypatch.setattr(client, "batch_put_ranges_from_ptr", blocking_put)
    return started


def test_stalled_store_stays_pending_and_refuses_new_transfers(server, monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    worker = _runtime(timeout_ms=20).create_worker_handle("ns", RankTopology(), None)
    source = torch.arange(16, dtype=torch.uint8)
    worker.register_buffers({"layer0": source})
    release = threading.Event()
    started = _stalled_store(server, monkeypatch, release)
    finished = threading.Event()

    try:
        job = worker.store([_block_plan("slow", source)])
        worker._futures[id(job)].add_done_callback(lambda _: finished.set())
        assert started.wait(5)
        clock.now += 1

        # MORI still reads the source, so the job must not settle as failed.
        assert worker.poll(job) is None
        for refused in (
            worker.store([_block_plan("next", source)]),
            worker.store_blocks([BlockTransferPlan("next", 0, group_id=0)]),
            worker.load([_block_plan("slow", source)]),
            worker.load_blocks([BlockTransferPlan("slow", 0, group_id=0)]),
        ):
            assert refused is not None
            for settle in (worker.wait, worker.cancel):
                assert settle(refused) is refused
            assert refused.status is TransferJobStatus.FAILED
            assert refused.error.startswith(TRANSFER_NOT_ATTEMPTED)

        release.set()
        assert finished.wait(5)
        # A finished store no longer blocks, even before it is polled.
        after = worker.wait(worker.store([_block_plan("after", source)]))
        assert after.status is TransferJobStatus.COMPLETED
        loaded = worker.wait(worker.load([_block_plan("after", source)]))
        assert loaded.status is TransferJobStatus.COMPLETED
        result = worker.poll(job)
        assert result is not None and result.status is TransferJobStatus.COMPLETED
        assert {"slow", "after"} <= server.objects.keys()
        assert not (worker._futures or worker._store_deadlines)
        assert not worker._load_deadlines
    finally:
        release.set()
        worker.close()
