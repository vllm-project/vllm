# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ValkeyL2Adapter.

These tests exercise the L2AdapterInterface contract against an
in-process fake of ``glide_sync`` so neither ``valkey-glide`` nor a real
Valkey server is required.
"""

# Standard
from typing import Any, Optional
import select
import sys
import threading
import time
import types

# Third Party
import pytest
import torch

# ---------------------------------------------------------------------------
# In-process fake `glide_sync`
# ---------------------------------------------------------------------------
#
# The worker pool imports ``glide_sync`` lazily when it creates a client,
# so the fake only has to be in ``sys.modules`` while a test is running --
# importing the adapter module does not pull glide in.  The ``_fake_glide``
# autouse fixture below installs it per test and removes it afterwards.
# Installing it at import time instead would leave the fake in
# ``sys.modules`` for the rest of the session, and
# ``test_valkey_l2_adapter_integration.py`` (collected after this file)
# would then "connect" to the in-process fake and pass without ever
# touching a real server.  All workers share the same backing dict so
# multi-thread behavior matches a real centralized server.


class MockValkeyServer:
    """Shared backing state for the fake glide clients — a single
    in-process stand-in for a Valkey server.

    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.store: dict[bytes, bytes] = {}
        # operation ("set"/"get"/"exists"/"delete") -> an Exception to
        # raise on the next call, or a callable
        # (key: bytes) -> Optional[Exception] for per-key faults.
        self.faults: dict[str, object] = {}
        # When set, ``set`` writes only the first ``truncate_bytes`` bytes
        # of the value — simulates a stale / incompatible entry.
        self.truncate_bytes: Optional[int] = None
        # Per-node INFO memory the cluster client's ``info()`` returns
        # (node addr -> INFO bytes).
        self.node_info: dict[str, bytes] = {}

    def reset(self) -> None:
        with self.lock:
            self.store.clear()
        self.faults.clear()
        self.truncate_bytes = None
        self.node_info.clear()

    def maybe_fault(self, operation: str, key: bytes) -> None:
        """Raise the fault configured for ``operation`` (one of
        ``"set"`` / ``"get"`` / ``"exists"`` / ``"delete"``), if any."""
        fault = self.faults.get(operation)
        if fault is None:
            return
        if callable(fault):
            result = fault(key)
            if result is not None:
                raise result
        elif isinstance(fault, BaseException):
            raise fault


_SERVER = MockValkeyServer()


class _FakeGlideClient:
    """Minimal in-process stand-in for ``glide_sync.GlideClient``."""

    def __init__(self) -> None:
        self.closed = False

    @classmethod
    def create(cls, config: object) -> "_FakeGlideClient":
        inst = cls()
        inst.config = config  # type: ignore[attr-defined]
        return inst

    def set(self, key: bytes, value) -> None:
        _SERVER.maybe_fault("set", bytes(key))
        v = bytes(value)
        if _SERVER.truncate_bytes is not None:
            v = v[: _SERVER.truncate_bytes]
        with _SERVER.lock:
            _SERVER.store[bytes(key)] = v

    def get(self, key: bytes, buffer=None):
        _SERVER.maybe_fault("get", bytes(key))
        with _SERVER.lock:
            v = _SERVER.store.get(bytes(key))
        if v is None:
            return None
        if buffer is None:
            return v
        n = min(len(v), len(buffer))
        buffer[:n] = v[:n]
        return n  # buffer GET returns bytes-written

    def exists(self, keys) -> int:
        for k in keys:
            _SERVER.maybe_fault("exists", bytes(k))
        with _SERVER.lock:
            return sum(1 for k in keys if bytes(k) in _SERVER.store)

    def delete(self, keys) -> int:
        for k in keys:
            _SERVER.maybe_fault("delete", bytes(k))
        n = 0
        with _SERVER.lock:
            for k in keys:
                kb = bytes(k)
                if kb in _SERVER.store:
                    del _SERVER.store[kb]
                    n += 1
        return n

    def close(self) -> None:
        self.closed = True


class _FakeGlideClusterClient(_FakeGlideClient):
    """Cluster client behaves identically against the shared fake, plus a
    stubbed ``info()`` that returns ``_SERVER.node_info`` for AllNodes
    routing."""

    def info(self, sections=None, route=None):
        # AllNodes route → dict of node addr -> INFO bytes.
        return dict(_SERVER.node_info)


def _build_fake_glide_modules() -> dict[str, types.ModuleType]:
    """Build the fake ``glide_sync`` / ``glide_shared`` module objects.

    Returns:
        A mapping of module name to module object, ready to be spliced
        into ``sys.modules`` for the duration of a test.
    """
    fake = types.ModuleType("glide_sync")

    def _record(name):
        def _ctor(**kw):
            return (name, kw)

        return _ctor

    fake.ServerCredentials = lambda u, p: ("creds", u, p)  # type: ignore[attr-defined]
    fake.NodeAddress = lambda h, p: ("addr", h, p)  # type: ignore[attr-defined]
    fake.AdvancedGlideClientConfiguration = _record("adv_std")  # type: ignore[attr-defined]
    fake.AdvancedGlideClusterClientConfiguration = _record(  # type: ignore[attr-defined]
        "adv_cluster"
    )
    fake.GlideClientConfiguration = _record("cfg_std")  # type: ignore[attr-defined]
    fake.GlideClusterClientConfiguration = _record("cfg_cluster")  # type: ignore[attr-defined]
    fake.GlideClient = _FakeGlideClient  # type: ignore[attr-defined]
    fake.GlideClusterClient = _FakeGlideClusterClient  # type: ignore[attr-defined]

    # Stub the glide_shared modules that `_do_node_memory` lazily imports
    # for per-node INFO routing.
    routes_mod = types.ModuleType("glide_shared.routes")
    routes_mod.AllNodes = lambda: ("route", "all_nodes")  # type: ignore[attr-defined]
    core_opts_mod = types.ModuleType("glide_shared.commands.core_options")

    class _InfoSection:
        MEMORY = "memory"

    core_opts_mod.InfoSection = _InfoSection  # type: ignore[attr-defined]
    return {
        "glide_sync": fake,
        "glide_shared": types.ModuleType("glide_shared"),
        "glide_shared.commands": types.ModuleType("glide_shared.commands"),
        "glide_shared.commands.core_options": core_opts_mod,
        "glide_shared.routes": routes_mod,
    }


# First Party
from lmcache.v1.distributed.api import (  # noqa: E402
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L2AdapterListener  # noqa: E402
from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (  # noqa: E402
    ValkeyL2Adapter,
    ValkeyL2AdapterConfig,
    _parse_startup_nodes,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _RecordingListener(L2AdapterListener):
    """Captures listener events for inspection in tests."""

    def __init__(self) -> None:
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []
        self.lock = threading.Lock()

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        with self.lock:
            self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        with self.lock:
            self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        with self.lock:
            self.deleted.append(list(keys))


def create_object_key(
    chunk_id: int,
    model_name: str = "test_model",
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        cache_salt=cache_salt,
    )


def create_memory_obj(size: int = 64, fill_value: float = 1.0) -> TensorMemoryObj:
    raw = torch.empty(size, dtype=torch.float32)
    raw.fill_(fill_value)
    meta = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _in_store(adapter: ValkeyL2Adapter, key: ObjectKey) -> bool:
    """Whether ``key`` currently exists in the fake Valkey store."""
    return adapter._wire_key(key).encode() in _SERVER.store  # noqa: SLF001


def wait_for_event_fd(fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    if not poll.poll(timeout * 1000):
        return False
    try:
        consume_fd(fd)
    except BlockingIOError:
        pass
    return True


def _wait_for_store(adapter: ValkeyL2Adapter, task_id: int, timeout: float = 5.0):
    """Poll the store event fd until ``task_id`` shows up; return its result."""
    fd = adapter.get_store_event_fd()
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    # Drain may report extra completions; loop until task_id appears, but
    # never past the wall-clock deadline the caller asked for.
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not poll.poll(remaining * 1000):
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        completed = adapter.pop_completed_store_tasks()
        if task_id in completed:
            return completed[task_id]
    raise AssertionError(f"store task {task_id} did not complete in {timeout}s")


def _wait_for_lookup(adapter: ValkeyL2Adapter, task_id: int, timeout: float = 5.0):
    fd = adapter.get_lookup_and_lock_event_fd()
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not poll.poll(remaining * 1000):
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        bm = adapter.query_lookup_and_lock_result(task_id)
        if bm is not None:
            return bm
    raise AssertionError(f"lookup task {task_id} did not complete in {timeout}s")


def _wait_for_load(adapter: ValkeyL2Adapter, task_id: int, timeout: float = 5.0):
    fd = adapter.get_load_event_fd()
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not poll.poll(remaining * 1000):
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        bm = adapter.query_load_result(task_id)
        if bm is not None:
            return bm
    raise AssertionError(f"load task {task_id} did not complete in {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_glide(monkeypatch):
    """Make the worker pool's lazy ``import glide_sync`` resolve to the fake.

    Scoped to a single test: ``monkeypatch.setitem`` restores (or removes)
    each ``sys.modules`` entry on teardown, so the fake never survives into
    the real-server integration tests in this session.
    """
    for name, module in _build_fake_glide_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    yield


@pytest.fixture(autouse=True)
def _reset_state():
    _SERVER.reset()
    yield
    _SERVER.reset()


def _make_config(**overrides: Any) -> ValkeyL2AdapterConfig:
    base: dict[str, Any] = {
        "startup_nodes": [("localhost", 6379)],
        "num_workers": 2,
        "connection_timeout": 2.0,
        "request_timeout": 2.0,
    }
    base.update(overrides)
    return ValkeyL2AdapterConfig(**base)


@pytest.fixture
def adapter():
    a = ValkeyL2Adapter(_make_config())
    yield a
    a.close()


@pytest.fixture
def cluster_adapter():
    a = ValkeyL2Adapter(_make_config(cluster_mode=True))
    yield a
    a.close()


# ===========================================================================
# Config validation
# ===========================================================================


class TestConfigValidation:
    def test_empty_startup_nodes_rejected(self):
        with pytest.raises(ValueError, match="startup_nodes"):
            ValkeyL2AdapterConfig(startup_nodes=[])

    def test_bad_port_rejected(self):
        with pytest.raises(ValueError):
            ValkeyL2AdapterConfig(startup_nodes=[("h", 0)])

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError, match="max_capacity_gb"):
            ValkeyL2AdapterConfig(startup_nodes=[("h", 1)], max_capacity_gb=-1)

    def test_zero_workers_rejected(self):
        with pytest.raises(ValueError, match="num_workers"):
            ValkeyL2AdapterConfig(startup_nodes=[("h", 1)], num_workers=0)

    def test_at_sign_in_prefix_rejected(self):
        with pytest.raises(ValueError, match="key_prefix"):
            ValkeyL2AdapterConfig(startup_nodes=[("h", 1)], key_prefix="bad@prefix")

    def test_cluster_mode_warns_on_database_id(self, caplog):
        # Standard
        import logging

        with caplog.at_level(logging.WARNING):
            cfg = ValkeyL2AdapterConfig(
                startup_nodes=[("h", 1)],
                cluster_mode=True,
                database_id=3,
            )
        assert cfg.database_id is None

    def test_standalone_warns_on_multiple_nodes(self, caplog):
        # Standard
        import logging

        with caplog.at_level(logging.WARNING):
            ValkeyL2AdapterConfig(
                startup_nodes=[("h1", 1), ("h2", 2)],
                cluster_mode=False,
            )


class TestParseStartupNodes:
    def test_single(self):
        assert _parse_startup_nodes("host:6379") == [("host", 6379)]

    def test_comma_separated(self):
        nodes = _parse_startup_nodes("a:1,b:2,c:3")
        assert nodes == [("a", 1), ("b", 2), ("c", 3)]

    def test_missing_colon(self):
        with pytest.raises(ValueError, match="host:port"):
            _parse_startup_nodes("nocolon")

    def test_non_integer_port(self):
        with pytest.raises(ValueError, match="non-integer port"):
            _parse_startup_nodes("host:abc")

    def test_empty_or_non_string_rejected(self):
        for bad in ("", "   ", None, [("a", 1)]):
            with pytest.raises(ValueError):
                _parse_startup_nodes(bad)


class TestFromDict:
    def test_basic(self):
        cfg = ValkeyL2AdapterConfig.from_dict(
            {
                "type": "valkey",
                "startup_nodes": "a:1,b:2",
                "cluster_mode": True,
                "username": "u",
                "password": "p",
                "key_prefix": "deploy1",
                "num_workers": 4,
                "tls_enable": True,
                "max_capacity_gb": 2.5,
            }
        )
        assert cfg.startup_nodes == [("a", 1), ("b", 2)]
        assert cfg.cluster_mode is True
        assert cfg.key_prefix == "deploy1"
        assert cfg.num_workers == 4
        assert cfg.tls_enable is True
        assert cfg.max_capacity_gb == 2.5


class TestStore:
    def test_single_key_round_trip(self, adapter):
        k = create_object_key(1)
        o = create_memory_obj(size=16, fill_value=0.5)
        task = adapter.submit_store_task([k], [o])
        result = _wait_for_store(adapter, task)
        assert result.is_successful()
        assert result.bytes_transferred() == o.get_size()

    def test_empty_batch_completes_immediately(self, adapter):
        task = adapter.submit_store_task([], [])
        result = _wait_for_store(adapter, task)
        assert result.is_successful()
        assert result.bytes_transferred() == 0

    def test_partial_failure_accounting(self, adapter):
        """
        partial batch failure must report the
        task as NOT successful and  account
        only the keys that actually wrote in per-salt
        """
        listener = _RecordingListener()
        adapter.register_listener(listener)
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(size=16) for _ in range(3)]

        # Make the SET for key index 1's wire key fail.
        target_wire = adapter._wire_key(keys[1]).encode()  # noqa: SLF001

        def faulty(k: bytes):
            if k == target_wire:
                return RuntimeError("simulated SET failure")
            return None

        _SERVER.faults["set"] = faulty

        task = adapter.submit_store_task(keys, objs)
        result = _wait_for_store(adapter, task)

        # Task-level: a partial failure is a task failure.
        assert not result.is_successful()
        assert result.bytes_transferred() == 0

        # Real accounting: only the 2 successful keys are counted.
        usage = adapter.get_usage()
        assert usage.total_bytes_used == 2 * objs[0].get_size()
        # verify the failed key is not in the stored-listener notifications.
        stored_flat = {k for batch in listener.stored for k in batch}
        assert keys[0] in stored_flat
        assert keys[2] in stored_flat
        assert keys[1] not in stored_flat

    def test_length_mismatch_raises(self, adapter):
        with pytest.raises(ValueError, match="length mismatch"):
            adapter.submit_store_task([create_object_key(0)], [])

    def test_store_fires_listener(self, adapter):
        listener = _RecordingListener()
        adapter.register_listener(listener)
        keys = [create_object_key(i) for i in range(2)]
        objs = [create_memory_obj(size=8) for _ in keys]
        task = adapter.submit_store_task(keys, objs)
        _wait_for_store(adapter, task)
        # ``_notify_keys_stored`` should have fired with these keys.
        assert any(set(batch) == set(keys) for batch in listener.stored)


class TestLookupAndLock:
    def test_lookup_after_store(self, adapter):
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(size=8) for _ in keys]
        _wait_for_store(adapter, adapter.submit_store_task(keys, objs))

        task = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        bm = _wait_for_lookup(adapter, task)
        assert all(bm.test(i) for i in range(3))

    def test_lookup_miss(self, adapter):
        keys = [create_object_key(99)]
        bm = _wait_for_lookup(
            adapter, adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        )
        assert not bm.test(0)

    def test_unlock_balances_lookup(self, adapter):
        keys = [create_object_key(i) for i in range(2)]
        objs = [create_memory_obj(size=4) for _ in keys]
        _wait_for_store(adapter, adapter.submit_store_task(keys, objs))
        # Two successful lookups → refcount == 2 per key.
        for _ in range(2):
            _wait_for_lookup(
                adapter, adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
            )
        # Two unlocks should bring it back to zero — no internal state
        # to assert publicly, just verify no exception and that we can
        # delete the keys afterward.
        adapter.submit_unlock(keys)
        adapter.submit_unlock(keys)
        adapter.delete(keys)


class TestLoad:
    def test_load_after_store_returns_hit(self, adapter):
        k = create_object_key(1)
        src = create_memory_obj(size=8, fill_value=0.25)
        _wait_for_store(adapter, adapter.submit_store_task([k], [src]))

        dst = create_memory_obj(size=8, fill_value=0.0)
        bm = _wait_for_load(adapter, adapter.submit_load_task([k], [dst]))
        assert bm.test(0)
        # Loaded buffer matches the source data.
        assert torch.allclose(dst.tensor, src.tensor)

    def test_load_miss_returns_zero_bit(self, adapter):
        dst = create_memory_obj(size=8)
        bm = _wait_for_load(
            adapter, adapter.submit_load_task([create_object_key(42)], [dst])
        )
        assert not bm.test(0)

    def test_size_mismatch_treated_as_miss(self, adapter):
        """A stale/wrong-size GET must be reported as a cache miss."""
        k = create_object_key(1)
        obj = create_memory_obj(size=16)  # 64 bytes (16 * float32)

        # Simulate a stale entry of the wrong length on the server.
        _SERVER.truncate_bytes = 8  # store only 8 bytes
        _wait_for_store(adapter, adapter.submit_store_task([k], [obj]))
        _SERVER.truncate_bytes = None

        dst = create_memory_obj(size=16)
        bm = _wait_for_load(adapter, adapter.submit_load_task([k], [dst]))
        assert not bm.test(0), "size-mismatched value must be reported as miss"

    def test_length_mismatch_raises(self, adapter):
        with pytest.raises(ValueError, match="length mismatch"):
            adapter.submit_load_task([create_object_key(0)], [])


class TestDelete:
    def test_delete_after_store(self, adapter):
        listener = _RecordingListener()
        adapter.register_listener(listener)
        keys = [create_object_key(i) for i in range(2)]
        objs = [create_memory_obj(size=8) for _ in keys]
        _wait_for_store(adapter, adapter.submit_store_task(keys, objs))

        adapter.delete(keys)
        # Listener observed the deletions.
        assert any(set(batch) == set(keys) for batch in listener.deleted)
        # Subsequent lookup should miss.
        bm = _wait_for_lookup(
            adapter, adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        )
        assert not any(bm.test(i) for i in range(len(keys)))

    def test_delete_unknown_keys_is_noop(self, adapter):
        # Should not raise even though the keys were never stored.
        adapter.delete([create_object_key(999)])

    def test_lock_blocks_delete(self, adapter):
        """A key pinned by an in-flight lookup must not be deleted."""
        key = create_object_key(1)
        _wait_for_store(
            adapter, adapter.submit_store_task([key], [create_memory_obj(4)])
        )

        # Lookup bumps the lock refcount.
        bm = _wait_for_lookup(
            adapter, adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        )
        assert bm.test(0)

        adapter.delete([key])
        assert _in_store(adapter, key), "locked key must survive delete"

        # After unlock the key is deletable again.
        adapter.submit_unlock([key])
        adapter.delete([key])
        assert not _in_store(adapter, key)

    def test_refcount_blocks_until_fully_unlocked(self, adapter):
        """Two lookups → refcount 2 → needs two unlocks before delete."""
        key = create_object_key(1)
        _wait_for_store(
            adapter, adapter.submit_store_task([key], [create_memory_obj(4)])
        )
        for _ in range(2):
            _wait_for_lookup(
                adapter, adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
            )

        adapter.submit_unlock([key])  # refcount 1, still pinned
        adapter.delete([key])
        assert _in_store(adapter, key)

        adapter.submit_unlock([key])  # refcount 0
        adapter.delete([key])
        assert not _in_store(adapter, key)


class TestKeyNamespacing:
    def test_different_prefixes_do_not_collide(self):
        a1 = ValkeyL2Adapter(_make_config(key_prefix="dep-A"))
        a2 = ValkeyL2Adapter(_make_config(key_prefix="dep-B"))
        try:
            k = create_object_key(7)
            obj = create_memory_obj(size=4)
            _wait_for_store(a1, a1.submit_store_task([k], [obj]))
            # a2 sees no value for the same logical key.
            bm = _wait_for_lookup(
                a2, a2.submit_lookup_and_lock_task([k], _EMPTY_LAYOUT)
            )
            assert not bm.test(0)
        finally:
            a1.close()
            a2.close()

    def test_cache_salt_isolation(self, adapter):
        # cache_salt is part of the wire key produced by
        # _object_key_to_string, so different salts must miss.
        k_a = create_object_key(1, cache_salt="user-A")
        k_b = create_object_key(1, cache_salt="user-B")
        obj = create_memory_obj(size=4)
        _wait_for_store(adapter, adapter.submit_store_task([k_a], [obj]))
        bm = _wait_for_lookup(
            adapter, adapter.submit_lookup_and_lock_task([k_b], _EMPTY_LAYOUT)
        )
        assert not bm.test(0)
