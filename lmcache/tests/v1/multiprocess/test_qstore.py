# SPDX-License-Identifier: Apache-2.0
"""Unit tests for QStoreModule: the server side of query-tensor transfer.
Covers Q ring registration/liveness/reaping, IPC release ordering, handler and
protocol wiring, and store_q's fail-closed guards.
"""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
import contextlib
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.modules.experimental import TRANSFER_QUERY
from lmcache.v1.multiprocess.modules.experimental import qstore as qstore_mod
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import ContextEntry

REGISTER_ARGS = ("model##query", 2)


def _ctx() -> MagicMock:
    """Creates a context stub to test QStoreModule."""
    ctx = MagicMock(name="ctx")
    ctx.chunk_size = 256
    ctx.separate_object_groups = False
    ctx.full_sw_kv = False
    return ctx


def _module(ctx: MagicMock | None = None) -> QStoreModule:
    """Creates a QStoreModule with a stub context."""
    return QStoreModule(ctx or _ctx())


def _stub_registration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Creates a stub for the create_cache_context call to test
    register_q_cache()."""
    create = MagicMock(return_value=MagicMock(num_layers=2))
    monkeypatch.setattr(qstore_mod, "create_cache_context", create)
    monkeypatch.setattr(qstore_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    return create


def _register(module: QStoreModule, instance_id: int = 1) -> None:
    """Registers a Q ring with the module, using the stubbed
    create_cache_context."""
    module.register_q_cache(
        instance_id, MagicMock(), *REGISTER_ARGS, MagicMock(), MagicMock(), []
    )


def test_register_inserts_unlatched_entry(monkeypatch) -> None:
    """Checks that successful registered entries inserts an entry to be
    tracked."""
    _stub_registration(monkeypatch)
    module = _module()

    _register(module)

    assert module.tracked_instance_count() == 1
    entry = module.get_and_touch_context_entry(1)
    assert entry is not None


def test_duplicate_register_refreshes_without_rebuilding(monkeypatch) -> None:
    """Checks that a duplicate registration refreshes the last_seen timestamp and
    does not create a second context or layout descriptor."""
    create = _stub_registration(monkeypatch)
    module = _module()
    _register(module)
    entry = module.get_and_touch_context_entry(1)
    assert entry is not None
    entry.last_seen = 0.0

    _register(module)

    assert create.call_count == 1
    assert module.tracked_instance_count() == 1
    assert entry.last_seen > 0.0


def test_reap_uses_two_tier_windows() -> None:
    """Check that the reaper uses two time windows to decide which entries to drop.
    This is a verbatim copy from tests/v1/multiprocess/test_worker_liveness.py"""
    module = _module()
    old = time.monotonic() - 1000.0
    module._q_contexts.update(
        {
            1: ContextEntry(MagicMock(), *REGISTER_ARGS, old, True),
            2: ContextEntry(MagicMock(), *REGISTER_ARGS, old, False),
            3: ContextEntry(MagicMock(), *REGISTER_ARGS, time.monotonic(), True),
        }
    )

    assert module.reap_stale_instances(120.0, 3600.0) == [1]
    assert module.tracked_instance_count() == 2
    cast(
        MagicMock, module.context.layout_desc_registry.unregister
    ).assert_called_once_with(*REGISTER_ARGS)


def test_unregister_releases_context_and_layout(monkeypatch) -> None:
    """Check that unregister_q_cache releases the context and layout descriptor, and
    that it tolerates repeated calls for the same instance."""
    calls: list[str] = []
    monkeypatch.setattr(
        qstore_mod,
        "torch_dev",
        SimpleNamespace(
            empty_cache=lambda: calls.append("empty_cache"),
            ipc_collect=lambda: calls.append("ipc_collect"),
        ),
    )
    module = _module()
    context = MagicMock()
    context.close.side_effect = lambda: calls.append("close")
    module._q_contexts[1] = ContextEntry(context, *REGISTER_ARGS, time.monotonic())

    module.unregister_q_cache(1)

    assert module.tracked_instance_count() == 0
    # Reclaim must come after close so the last IPC tensor reference is gone
    # before ipc_collect tries to unmap the segment (LMCache#4014).
    assert calls == ["close", "empty_cache", "ipc_collect"]
    cast(
        MagicMock, module.context.layout_desc_registry.unregister
    ).assert_called_once_with(*REGISTER_ARGS)

    module.unregister_q_cache(1)  # already gone -> no exception


def test_release_tolerates_backends_without_ipc_collect(monkeypatch) -> None:
    """xpu/musa device modules expose no ipc_collect; release must not fail."""
    monkeypatch.setattr(
        qstore_mod, "torch_dev", SimpleNamespace(empty_cache=lambda: None)
    )
    module = _module()
    module._q_contexts[1] = ContextEntry(MagicMock(), *REGISTER_ARGS, time.monotonic())

    module.unregister_q_cache(1)

    assert module.tracked_instance_count() == 0


class _FakeEvent:
    """Device event double exposing the record/ipc_handle surface store_q uses."""

    created: list["_FakeEvent"] = []

    def __init__(self, interprocess: bool = False) -> None:
        self.records = 0
        _FakeEvent.created.append(self)

    def record(self) -> None:
        self.records += 1

    def ipc_handle(self) -> bytes:
        return b"event-handle"

    def wait(self, stream=None) -> None:
        return None

    @classmethod
    def from_ipc_handle(cls, device, handle: bytes) -> "_FakeEvent":
        return cls()


@pytest.fixture
def stub_device(monkeypatch):
    """Replace the device module so store_q runs on CPU."""

    @contextlib.contextmanager
    def _null(*args, **kwargs):
        yield

    _FakeEvent.created.clear()
    monkeypatch.setattr(
        qstore_mod,
        "torch_dev",
        SimpleNamespace(device=_null, stream=_null, Event=_FakeEvent),
    )
    monkeypatch.setattr(qstore_mod, "check_interprocess_event_support", lambda: None)
    return _FakeEvent.created


def test_store_q_unregistered_instance_raises() -> None:
    """A store for an unknown instance is a protocol error."""
    with pytest.raises(ValueError, match="No Q ring registered"):
        _module().store_q(MagicMock(), 42, [[0]], b"handle")


def test_store_q_block_id_underflow_fails_closed(stub_device) -> None:
    """Short block-id lists would drive the transfer kernel out of bounds, so
    the whole store is skipped: nothing reserved, failure reported, event still
    recorded so the waiting worker is released."""
    ctx = _ctx()
    ctx.resolve_obj_keys.return_value = [["obj-0", "obj-1"]]  # 2 chunks
    module = _module(ctx)
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager.num_object_groups = 1
    cache_context.kv_layer_groups_manager.num_kernel_groups = 1
    cache_context.calculate_num_blocks.return_value = 2  # 2 chunks * 2 = 4 needed
    module._q_contexts[1] = ContextEntry(
        cache_context, *REGISTER_ARGS, time.monotonic()
    )

    handle, ok = module.store_q(MagicMock(), 1, [[0, 1, 2]], b"peer-handle")

    assert ok is False
    assert handle == b"event-handle"
    assert stub_device[0].records == 1
    cast(MagicMock, ctx.storage_manager.reserve_write).assert_not_called()
    cast(MagicMock, ctx.event_bus.publish).assert_not_called()


class _FakeLMCacheDriven:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakeEngineDriven:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakeQStore:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


@pytest.fixture
def stub_server_modules(monkeypatch):
    """Stub the server's module constructors. Returns the ManagementModule mock.
    The transfer modules stay real classes: _build_modules isinstance-checks
    them to pick liveness targets and the lmcache-driven module."""
    monkeypatch.setattr(server_mod, "LookupModule", lambda ctx: MagicMock())
    monkeypatch.setattr(server_mod, "P2PController", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(server_mod, "LMCacheDrivenTransferModule", _FakeLMCacheDriven)
    monkeypatch.setattr(server_mod, "EngineDrivenTransferModule", _FakeEngineDriven)
    monkeypatch.setattr(server_mod, "QStoreModule", _FakeQStore)
    management = MagicMock(name="ManagementModule")
    monkeypatch.setattr(server_mod, "ManagementModule", management)
    return management


def _build(stub_server_modules, **config) -> list:
    return server_mod._build_modules(
        MagicMock(name="ctx"), MPServerConfig(**config), MagicMock(url="")
    )


def test_server_rejects_an_unknown_experimental_module(stub_server_modules) -> None:
    """Check that the server rejects an unknown experimental module.
    Only 'transfer_query' is supported right now."""
    with pytest.raises(ValueError, match="Unknown --enable"):
        _build(stub_server_modules, enable=["query"])


def test_server_requires_lmcache_driven_transfer(stub_server_modules) -> None:
    """Check that the server rejects a request to enable transfer_query when the
    transfer mode is not lmcache-driven."""
    with pytest.raises(ValueError, match="lmcache_driven"):
        _build(
            stub_server_modules,
            enable=[TRANSFER_QUERY],
            supported_transfer_mode="engine_driven",
        )


def test_server_builds_q_store_module(stub_server_modules) -> None:
    """Check that the server builds the Q store module and puts it to the
    ManagementModule."""
    modules = _build(
        stub_server_modules,
        enable=[TRANSFER_QUERY],
        supported_transfer_mode="lmcache_driven",
    )

    assert any(isinstance(m, _FakeQStore) for m in modules)
    kwargs = stub_server_modules.call_args.kwargs
    assert kwargs["experimental_transfer"] == [TRANSFER_QUERY]
    assert any(isinstance(t, _FakeQStore) for t in kwargs["liveness_targets"])


def test_server_builds_nothing_when_no_feature_is_enabled(
    stub_server_modules,
) -> None:
    """Check that the server builds nothing when no feature is enabled."""
    modules = _build(stub_server_modules)

    assert not any(isinstance(m, _FakeQStore) for m in modules)
    assert stub_server_modules.call_args.kwargs["experimental_transfer"] == []
