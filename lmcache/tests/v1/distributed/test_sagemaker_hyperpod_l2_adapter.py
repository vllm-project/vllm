# SPDX-License-Identifier: Apache-2.0
"""Tests for the SageMaker HyperPod MP L2 adapter."""

# Standard
from multiprocessing import shared_memory
from typing import Iterator
import asyncio
import select
import socket
import threading
import time

# Third Party
from aiohttp import web
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.config import get_type_name_for_config
from lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_client import (
    SageMakerHyperPodClient,
    SageMakerHyperPodLease,
)
from lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_l2_adapter import (
    SageMakerHyperPodL2Adapter,
    SageMakerHyperPodL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


class _FakeHyperPodClient:
    """In-memory public-contract substitute for the ai-toolkit client."""

    instance: "_FakeHyperPodClient | None" = None

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.data: dict[str, bytes] = {}
        self.released: list[str] = []
        self.release_calls: list[str] = []
        self.fail_release = False
        self.lease_expires_in: float = float("inf")
        self.closed = False
        self.raise_acquire_keys: set[str] = set()
        self.block_put = False
        self.put_started = threading.Event()
        self.put_cancelled = False
        self._put_loop: asyncio.AbstractEventLoop | None = None
        self._put_gate: asyncio.Event | None = None
        type(self).instance = self

    @staticmethod
    def normalize_url(url: str, use_https: bool = False) -> str:
        return SageMakerHyperPodClient.normalize_url(url, use_https=use_https)

    def release_put(self) -> None:
        if self._put_loop is not None and self._put_gate is not None:
            self._put_loop.call_soon_threadsafe(self._put_gate.set)

    async def put(self, key: str, data: memoryview) -> bool:
        if self.block_put:
            self._put_loop = asyncio.get_running_loop()
            self._put_gate = asyncio.Event()
        self.put_started.set()
        try:
            if self._put_gate is not None:
                await self._put_gate.wait()
        except asyncio.CancelledError:
            self.put_cancelled = True
            raise
        self.data[key] = bytes(data)
        return True

    async def acquire_lease(self, key: str) -> SageMakerHyperPodLease | None:
        if key in self.raise_acquire_keys:
            raise RuntimeError(f"injected acquire failure for {key}")
        data = self.data.get(key)
        if data is None:
            return None
        return SageMakerHyperPodLease(
            key,
            ((0, len(data)),),
            expires_monotonic=time.monotonic() + self.lease_expires_in,
        )

    def copy_from_lease(
        self,
        lease: SageMakerHyperPodLease,
        destination: memoryview,
    ) -> bool:
        if lease.is_expired():
            return False
        data = self.data.get(lease.lease_id)
        view = destination.cast("B") if destination.format != "B" else destination
        if data is None or len(data) != len(view):
            return False
        view[:] = data
        return True

    async def release_lease(self, lease: SageMakerHyperPodLease) -> bool:
        self.release_calls.append(lease.lease_id)
        if self.fail_release:
            return False
        self.released.append(lease.lease_id)
        return True

    async def close(self) -> None:
        self.closed = True


def _memory_obj(elements: int = 16, fill: float = 1.0) -> TensorMemoryObj:
    raw = torch.full((elements,), fill, dtype=torch.float32)
    metadata = MemoryObjMetadata(
        shape=raw.shape,
        dtype=raw.dtype,
        address=raw.data_ptr(),
        phy_size=raw.numel() * raw.element_size(),
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw, metadata, parent_allocator=None)


def _key(value: int = 1, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(value),
        model_name="test/model",
        kv_rank=1,
        object_group_id=2,
        cache_salt=cache_salt,
    )


def _wait(fd: int, timeout: float = 2.0) -> None:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    assert poller.poll(int(timeout * 1000)), "adapter task did not complete"
    consume_fd(fd)


@pytest.fixture
def adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[SageMakerHyperPodL2Adapter]:
    # First Party
    import lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_l2_adapter as mod

    monkeypatch.setattr(mod, "SageMakerHyperPodClient", _FakeHyperPodClient)
    value = SageMakerHyperPodL2Adapter(
        SageMakerHyperPodL2AdapterConfig(
            url="sagemaker-hyperpod://127.0.0.1:9200",
        )
    )
    yield value
    value.close()


def test_config_registration_and_validation() -> None:
    config = SageMakerHyperPodL2AdapterConfig.from_dict(
        {
            "type": "sagemaker-hyperpod",
            "url": "sagemaker-hyperpod://127.0.0.1:9200",
        }
    )
    assert config.bucket == "lmcache"
    assert config.shared_memory_name == "shared_memory"
    assert get_type_name_for_config(config) == "sagemaker-hyperpod"

    with pytest.raises(ValueError, match="url"):
        SageMakerHyperPodL2AdapterConfig.from_dict(
            {"type": "sagemaker-hyperpod", "url": "127.0.0.1:9200"}
        )
    https_config = SageMakerHyperPodL2AdapterConfig.from_dict(
        {
            "type": "sagemaker-hyperpod",
            "url": "sagemaker-hyperpod://127.0.0.1:9200",
            "use_https": True,
        }
    )
    assert https_config.use_https is True
    assert (
        SageMakerHyperPodClient.normalize_url(
            "sagemaker-hyperpod://127.0.0.1:9200", use_https=True
        )
        == "https://127.0.0.1:9200"
    )

    with pytest.raises(ValueError, match="scheme"):
        SageMakerHyperPodL2AdapterConfig.from_dict(
            {"type": "sagemaker-hyperpod", "url": "http://127.0.0.1:9200"}
        )
    with pytest.raises(ValueError, match="eviction"):
        SageMakerHyperPodL2AdapterConfig.from_dict(
            {
                "type": "sagemaker-hyperpod",
                "url": "sagemaker-hyperpod://127.0.0.1:9200",
                "eviction": {"policy": "lru"},
            }
        )


def test_event_descriptors_are_distinct(adapter: SageMakerHyperPodL2Adapter) -> None:
    assert (
        len(
            {
                adapter.get_store_event_fd(),
                adapter.get_lookup_and_lock_event_fd(),
                adapter.get_load_event_fd(),
            }
        )
        == 3
    )


def test_store_lookup_load_and_unlock(adapter: SageMakerHyperPodL2Adapter) -> None:
    key = _key(cache_salt="tenant-a")
    source = _memory_obj(fill=3.0)
    destination = _memory_obj(fill=0.0)

    store_id = adapter.submit_store_task([key], [source])
    _wait(adapter.get_store_event_fd())
    store_result = adapter.pop_completed_store_tasks()[store_id]
    assert store_result.is_successful()
    assert store_result.bytes_transferred() == source.get_size()

    lookup_id = adapter.submit_lookup_and_lock_task([key, _key(99)], _EMPTY_LAYOUT)
    _wait(adapter.get_lookup_and_lock_event_fd())
    lookup = adapter.query_lookup_and_lock_result(lookup_id)
    assert lookup is not None
    assert lookup.test(0)
    assert not lookup.test(1)
    assert adapter.query_lookup_and_lock_result(lookup_id) is None

    load_id = adapter.submit_load_task([key], [destination])
    _wait(adapter.get_load_event_fd())
    loaded = adapter.query_load_result(load_id)
    assert loaded is not None and loaded.test(0)
    assert bytes(destination.byte_array) == bytes(source.byte_array)

    adapter.submit_unlock([key])
    client = _FakeHyperPodClient.instance
    assert client is not None
    deadline = time.monotonic() + 2
    while not client.released and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.released == ["test/model@00000001@2@00000001@tenant-a"]


def test_load_size_mismatch_is_a_miss(adapter: SageMakerHyperPodL2Adapter) -> None:
    key = _key()
    source = _memory_obj(elements=8)
    destination = _memory_obj(elements=16)

    store_id = adapter.submit_store_task([key], [source])
    _wait(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[store_id].is_successful()

    load_id = adapter.submit_load_task([key], [destination])
    _wait(adapter.get_load_event_fd())
    loaded = adapter.query_load_result(load_id)
    assert loaded is not None and not loaded.test(0)


def test_client_copies_fragmented_shared_memory() -> None:
    shm = shared_memory.SharedMemory(create=True, size=16)
    try:
        shm.buf[:8] = b"abcdefgh"
        client = SageMakerHyperPodClient(
            url="sagemaker-hyperpod://127.0.0.1:9200",
            shared_memory_name=shm.name,
        )
        destination = memoryview(bytearray(6))
        lease = SageMakerHyperPodLease("lease", ((1, 3), (5, 3)))
        assert client.copy_from_lease(lease, destination)
        assert bytes(destination) == b"bcdfgh"
        destination.release()

        # A destination whose size differs from the lease is a miss and is
        # never written.
        wrong_size = memoryview(bytearray(b"XXXX"))
        assert not client.copy_from_lease(lease, wrong_size)
        assert bytes(wrong_size) == b"XXXX"
        wrong_size.release()
        asyncio.run(client.close())
    finally:
        shm.close()
        shm.unlink()


def test_client_http_and_lease_protocol() -> None:
    async def scenario() -> None:
        state: dict[str, object] = {
            "put_status": 200,
            "stored": b"",
            "key": "",
            "released": [],
            "malformed_lease": False,
            "invalid_offsets": False,
        }

        async def put(request: web.Request) -> web.Response:
            state["key"] = request.match_info["key"]
            state["stored"] = await request.read()
            put_status = state["put_status"]
            assert isinstance(put_status, int)
            return web.Response(status=put_status)

        async def acquire(request: web.Request) -> web.Response:
            if state["malformed_lease"]:
                return web.json_response(["not", "an", "object"])
            if state["invalid_offsets"]:
                return web.json_response(
                    {"id": "lease/bad", "offsets": [{"offset": -1, "len": 4}]}
                )
            return web.json_response(
                {
                    "id": "lease/1",
                    "offsets": [{"offset": 2, "len": 4}],
                }
            )

        async def release(request: web.Request) -> web.Response:
            released = state["released"]
            assert isinstance(released, list)
            released.append(request.match_info["lease_id"])
            return web.Response(status=200)

        app = web.Application()
        app.router.add_put("/v1/kv/{bucket}/{key:.+}", put)
        app.router.add_post("/v1/kv/{bucket}/{key:.+}/leases", acquire)
        app.router.add_post("/v1/leases/{lease_id:.+}/release", release)
        runner = web.AppRunner(app)
        await runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        site = web.SockSite(runner, sock)
        await site.start()

        shm = shared_memory.SharedMemory(create=True, size=16)
        shm.buf[2:6] = b"data"
        client = SageMakerHyperPodClient(
            url=f"sagemaker-hyperpod://127.0.0.1:{port}",
            bucket="bucket name",
            shared_memory_name=shm.name,
        )
        try:
            key = "model/name@00000001@0@abcd"
            assert await client.put(key, memoryview(b"payload"))
            assert state["stored"] == b"payload"
            assert state["key"] == key

            state["put_status"] = 409
            assert await client.put(key, memoryview(b"payload"))
            state["put_status"] = 500
            assert not await client.put(key, memoryview(b"payload"))
            status = client.report_status()
            assert status["is_healthy"], "HTTP statuses must not trip the breaker"
            assert status["consecutive_http_failures"] == 0
            assert not status["circuit_open"]

            lease = await client.acquire_lease(key)
            assert lease is not None
            destination = memoryview(bytearray(4))
            assert client.copy_from_lease(lease, destination)
            assert bytes(destination) == b"data"
            destination.release()
            assert await client.release_lease(lease)
            assert state["released"] == ["lease/1"]
            assert client.report_status()["is_healthy"]

            state["malformed_lease"] = True
            assert await client.acquire_lease(key) is None

            state["malformed_lease"] = False
            state["invalid_offsets"] = True
            assert await client.acquire_lease(key) is None
            assert state["released"] == ["lease/1", "lease/bad"]
        finally:
            await client.close()
            shm.close()
            shm.unlink()
            await runner.cleanup()

    asyncio.run(scenario())


class _RecordingListener(L2AdapterListener):
    def __init__(self) -> None:
        self.stored: list[list[ObjectKey]] = []

    def on_l2_keys_stored(
        self,
        keys: list[ObjectKey],
        sizes: list[int],
    ) -> None:
        del sizes
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        del keys

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        del keys


def test_invalid_later_fragment_does_not_modify_destination() -> None:
    shm = shared_memory.SharedMemory(create=True, size=16)
    try:
        client = SageMakerHyperPodClient(
            url="sagemaker-hyperpod://127.0.0.1:9200",
            shared_memory_name=shm.name,
        )
        destination = memoryview(bytearray(b"XXXXXX"))
        lease = SageMakerHyperPodLease("lease", ((1, 3), (1 << 30, 3)))
        assert not client.copy_from_lease(lease, destination)
        assert bytes(destination) == b"XXXXXX"
        destination.release()
        asyncio.run(client.close())
    finally:
        shm.close()
        shm.unlink()


def test_lookup_partial_exception_preserves_hits(
    adapter: SageMakerHyperPodL2Adapter,
) -> None:
    existing = _key(1)
    failing = _key(99)
    source = _memory_obj()

    store_id = adapter.submit_store_task([existing], [source])
    _wait(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[store_id].is_successful()

    client = _FakeHyperPodClient.instance
    assert client is not None
    client.raise_acquire_keys.add("test/model@00000001@2@00000063")

    lookup_id = adapter.submit_lookup_and_lock_task(
        [existing, failing],
        _EMPTY_LAYOUT,
    )
    _wait(adapter.get_lookup_and_lock_event_fd())
    result = adapter.query_lookup_and_lock_result(lookup_id)
    assert result is not None
    assert result.test(0)
    assert not result.test(1)

    adapter.submit_unlock([existing])
    deadline = time.monotonic() + 2
    while not client.released and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.released == ["test/model@00000001@2@00000001"]


def test_submit_store_snapshots_keys(
    adapter: SageMakerHyperPodL2Adapter,
) -> None:
    original = _key(1)
    replacement = _key(2)
    keys = [original]
    listener = _RecordingListener()
    adapter.register_listener(listener)

    client = _FakeHyperPodClient.instance
    assert client is not None
    client.block_put = True

    task_id = adapter.submit_store_task(keys, [_memory_obj()])
    assert client.put_started.wait(timeout=2)
    keys[0] = replacement
    client.release_put()

    _wait(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[task_id].is_successful()
    assert listener.stored == [[original]]


def test_close_cancels_inflight_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    import lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_l2_adapter as mod

    monkeypatch.setattr(mod, "SageMakerHyperPodClient", _FakeHyperPodClient)
    adapter = SageMakerHyperPodL2Adapter(
        SageMakerHyperPodL2AdapterConfig(
            url="sagemaker-hyperpod://127.0.0.1:9200",
            lease_ttl_ms=200,
        )
    )
    client = _FakeHyperPodClient.instance
    assert client is not None
    client.block_put = True

    adapter.submit_store_task([_key()], [_memory_obj()])
    assert client.put_started.wait(timeout=2)
    adapter.close()

    assert client.put_cancelled
    assert client.closed


def test_client_circuit_breaker_fast_fails_and_probes() -> None:
    async def scenario() -> None:
        # Reserve a port, then leave it closed so requests are refused.
        reserve = socket.socket()
        reserve.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
        reserve.close()

        shm = shared_memory.SharedMemory(create=True, size=8)
        client = SageMakerHyperPodClient(
            url=f"sagemaker-hyperpod://127.0.0.1:{port}",
            shared_memory_name=shm.name,
            timeout_ms=500,
        )
        client.circuit_cooldown_s = 0.2
        runner: web.AppRunner | None = None
        try:
            payload = memoryview(b"x")
            for _ in range(client.failure_threshold):
                assert not await client.put("key", payload)
            status = client.report_status()
            assert status["circuit_open"]
            assert not status["is_healthy"]
            failures = status["consecutive_http_failures"]
            assert isinstance(failures, int)
            assert failures == client.failure_threshold

            # Circuit open: fail fast without attempting (failure count
            # stays put because no request is issued).
            assert not await client.put("key", payload)
            status = client.report_status()
            assert status["consecutive_http_failures"] == failures

            # After the cooldown one probe goes through, fails against the
            # still-closed port, and increments the failure count.
            await asyncio.sleep(0.25)
            assert not await client.put("key", payload)
            status = client.report_status()
            assert status["consecutive_http_failures"] == failures + 1

            # Bring a daemon up on the reserved port: the next probe gets an
            # HTTP response (status irrelevant) and closes the circuit.
            async def put(request: web.Request) -> web.Response:
                await request.read()
                return web.Response(status=200)

            app = web.Application()
            app.router.add_put("/v1/kv/{bucket}/{key:.+}", put)
            runner = web.AppRunner(app)
            await runner.setup()
            recover_sock = socket.socket()
            recover_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            recover_sock.bind(("127.0.0.1", port))
            await web.SockSite(runner, recover_sock).start()

            await asyncio.sleep(0.25)
            assert await client.put("key", payload)
            status = client.report_status()
            assert status["is_healthy"]
            assert not status["circuit_open"]
            assert status["consecutive_http_failures"] == 0
        finally:
            await client.close()
            shm.close()
            shm.unlink()
            if runner is not None:
                await runner.cleanup()

    asyncio.run(scenario())


def test_close_releases_leases_once_without_ttl_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    import lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_l2_adapter as mod

    monkeypatch.setattr(mod, "SageMakerHyperPodClient", _FakeHyperPodClient)
    adapter = SageMakerHyperPodL2Adapter(
        SageMakerHyperPodL2AdapterConfig(
            url="sagemaker-hyperpod://127.0.0.1:9200",
            lease_ttl_ms=30000,
        )
    )
    key = _key()
    store_id = adapter.submit_store_task([key], [_memory_obj()])
    _wait(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[store_id].is_successful()

    lookup_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
    _wait(adapter.get_lookup_and_lock_event_fd())
    lookup = adapter.query_lookup_and_lock_result(lookup_id)
    assert lookup is not None and lookup.test(0)

    client = _FakeHyperPodClient.instance
    assert client is not None
    client.fail_release = True

    begin = time.monotonic()
    adapter.close()
    elapsed = time.monotonic() - begin
    assert elapsed < 5.0, "close must not retry lease release until the TTL"
    assert client.release_calls == ["test/model@00000001@2@00000001"]
    assert client.closed


def test_load_transient_acquire_exception_is_isolated(
    adapter: SageMakerHyperPodL2Adapter,
) -> None:
    good = _key(1)
    failing = _key(99)
    sources: list[MemoryObj] = [_memory_obj(fill=5.0), _memory_obj(fill=7.0)]
    destinations: list[MemoryObj] = [_memory_obj(fill=0.0), _memory_obj(fill=0.0)]

    store_id = adapter.submit_store_task([good, failing], sources)
    _wait(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[store_id].is_successful()

    client = _FakeHyperPodClient.instance
    assert client is not None
    client.raise_acquire_keys.add("test/model@00000001@2@00000063")

    # No prior lookup: both keys take the transient-lease path.
    load_id = adapter.submit_load_task([good, failing], destinations)
    _wait(adapter.get_load_event_fd())
    loaded = adapter.query_load_result(load_id)
    assert loaded is not None
    assert loaded.test(0)
    assert not loaded.test(1)
    assert bytes(destinations[0].byte_array) == bytes(sources[0].byte_array)


def test_copy_refuses_expired_lease() -> None:
    shm = shared_memory.SharedMemory(create=True, size=16)
    try:
        shm.buf[:8] = b"abcdefgh"
        client = SageMakerHyperPodClient(
            url="sagemaker-hyperpod://127.0.0.1:9200",
            shared_memory_name=shm.name,
        )
        destination = memoryview(bytearray(b"XXXX"))
        expired = SageMakerHyperPodLease(
            "lease",
            ((0, 4),),
            expires_monotonic=time.monotonic() - 1.0,
        )
        assert not client.copy_from_lease(expired, destination)
        assert bytes(destination) == b"XXXX", "destination must stay unmodified"

        valid = SageMakerHyperPodLease(
            "lease",
            ((0, 4),),
            expires_monotonic=time.monotonic() + 60.0,
        )
        assert client.copy_from_lease(valid, destination)
        assert bytes(destination) == b"abcd"
        destination.release()
        asyncio.run(client.close())
    finally:
        shm.close()
        shm.unlink()


def test_load_reacquires_when_retained_lease_expired(
    adapter: SageMakerHyperPodL2Adapter,
) -> None:
    key = _key()
    source = _memory_obj(fill=9.0)
    destination = _memory_obj(fill=0.0)

    store_id = adapter.submit_store_task([key], [source])
    _wait(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[store_id].is_successful()

    # The lookup retains a lease that is already past its TTL.
    client = _FakeHyperPodClient.instance
    assert client is not None
    client.lease_expires_in = 0.0

    lookup_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
    _wait(adapter.get_lookup_and_lock_event_fd())
    lookup = adapter.query_lookup_and_lock_result(lookup_id)
    assert lookup is not None and lookup.test(0)

    # The load must not read through the expired retained lease; it should
    # acquire a fresh (now valid) lease instead and still succeed.
    client.lease_expires_in = float("inf")

    load_id = adapter.submit_load_task([key], [destination])
    _wait(adapter.get_load_event_fd())
    loaded = adapter.query_load_result(load_id)
    assert loaded is not None and loaded.test(0)
    assert bytes(destination.byte_array) == bytes(source.byte_array)


def test_copy_discards_result_when_lease_expires_mid_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    import lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_client as client_mod

    shm = shared_memory.SharedMemory(create=True, size=8)
    try:
        shm.buf[:4] = b"data"
        client = SageMakerHyperPodClient(
            url="sagemaker-hyperpod://127.0.0.1:9200",
            shared_memory_name=shm.name,
        )

        # Deterministic clock: valid at the pre-copy check, expired by the
        # post-copy check — simulating a lease that dies mid-copy.
        base = time.monotonic()
        lease = SageMakerHyperPodLease("lease", ((0, 4),), expires_monotonic=base + 5.0)
        ticks = iter([base, base + 10.0])

        class _Clock:
            @staticmethod
            def monotonic() -> float:
                return next(ticks, base + 10.0)

        monkeypatch.setattr(client_mod, "time", _Clock)

        destination = memoryview(bytearray(4))
        assert not client.copy_from_lease(lease, destination), (
            "a copy finishing after the lease expiry must be discarded"
        )
        destination.release()
        monkeypatch.undo()
        asyncio.run(client.close())
    finally:
        shm.close()
        shm.unlink()


def test_client_attach_does_not_unlink_segment_on_process_exit() -> None:
    """Attach-only clients must never delete the daemon-owned segment.

    CPython's multiprocessing resource tracker registers even attach-only
    shared-memory segments and unlinks them at interpreter shutdown.
    Against ai-toolkit's host-owned arena that deletes the node-local
    cache for every client on the node (reproduced on HyperPod under
    ``hostIPC``). The client therefore attaches via a plain read-only
    ``mmap``, which involves no tracker: a client process exiting —
    even without calling ``close()`` — must leave the segment intact.
    """
    # Standard
    import os
    import subprocess
    import sys

    if not os.path.isdir("/dev/shm"):
        pytest.skip("requires /dev/shm (Linux)")

    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        shm.buf[:5] = b"hello"
        script = (
            "from lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_client "
            "import SageMakerHyperPodClient\n"
            "client = SageMakerHyperPodClient(\n"
            f"    url='sagemaker-hyperpod://127.0.0.1:1',\n"
            f"    shared_memory_name={shm.name!r},\n"
            ")\n"
            "assert client.report_status()['shared_memory_current'] is True\n"
            "# exit WITHOUT close(): the worst case for tracker cleanup\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        path = os.path.join("/dev/shm", shm.name.lstrip("/"))
        assert os.path.exists(path), (
            "client process exit deleted the daemon-owned segment "
            "(resource tracker regression)"
        )
    finally:
        shm.close()
        shm.unlink()


def test_client_arena_mapping_is_read_only_on_linux() -> None:
    """On Linux the segment mapping must be read-only: the client never
    writes to daemon-owned memory (writes go through the daemon's HTTP
    API).

    The read-only property is a memory-protection guarantee with no
    public write path by design, so this test exercises the mapping
    helper directly.
    """
    # Standard
    import os

    # First Party
    from lmcache.v1.distributed.l2_adapters.sagemaker_hyperpod_client import (
        _ReadOnlySharedMemoryMapping,
    )

    if not os.path.isdir("/dev/shm"):
        pytest.skip("requires /dev/shm (Linux)")

    shm = shared_memory.SharedMemory(create=True, size=64)
    try:
        shm.buf[:5] = b"hello"
        mapping = _ReadOnlySharedMemoryMapping(shm.name)
        try:
            assert mapping.size == 64
            assert bytes(mapping.buf[:5]) == b"hello"
            with pytest.raises(TypeError):
                mapping.buf[0:1] = b"x"
        finally:
            mapping.close()
    finally:
        shm.close()
        shm.unlink()
