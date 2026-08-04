# SPDX-License-Identifier: Apache-2.0
"""Unit tests for server-side worker liveness tracking and reaping.

Cover the public liveness interfaces of the transfer modules, the
management reaper wiring, the blend reap listener, and config validation.
No GPU or live server is required; module-level construction dependencies
are stubbed.
"""

# Standard
from typing import cast
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.modules import engine_driven_transfer as non_gpu_mod
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as gpu_mod
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.periodic_thread import PeriodicThreadRegistry


def _bare_gpu_module() -> LMCacheDrivenTransferModule:
    """A LMCacheDrivenTransferModule with only the liveness state initialized.

    Bypasses __init__ (which starts a CUDA host-func dispatcher) so the
    liveness methods can be exercised without GPU hardware.
    """
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = MagicMock(name="ctx")
    module._cache_contexts = {}
    module._lock = threading.Lock()
    return module


def _bare_non_gpu_module() -> EngineDrivenTransferModule:
    """A EngineDrivenTransferModule with only the liveness state initialized."""
    module = EngineDrivenTransferModule.__new__(EngineDrivenTransferModule)
    module._ctx = MagicMock(name="ctx")
    module._engine_driven_contexts = {}
    module._strategies = {}
    module._lock = threading.Lock()
    module._pending_shm_writes = {}
    module._pending_shm_reads = {}
    module._pending_shm_lock = threading.Lock()
    return module


def _stub_gpu_registration_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub Event IPC lookup for liveness tests unrelated to event handling."""
    monkeypatch.setattr(
        gpu_mod,
        "get_event_ipc_backend",
        MagicMock(return_value=MagicMock(name="event_backend")),
    )


def test_gpu_register_inserts_unlatched_entry(monkeypatch) -> None:
    """register_kv_cache inserts an entry that is not yet ping-proven."""
    _stub_gpu_registration_backend(monkeypatch)
    monkeypatch.setattr(
        gpu_mod, "create_cache_context", lambda *a, **kw: MagicMock(num_layers=2)
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    module = _bare_gpu_module()

    module.register_kv_cache(1, MagicMock(), "model", 1, MagicMock(), MagicMock(), [])

    assert module.tracked_instance_count() == 1
    entry = module.get_and_touch_context_entry(1)
    assert entry is not None and entry.has_liveness_signal is False


def test_gpu_noop_register_refreshes_without_latching(monkeypatch) -> None:
    """Re-registering a known instance refreshes last_seen but does not
    rebuild the context or latch the ping-proven flag."""
    _stub_gpu_registration_backend(monkeypatch)
    create = MagicMock(return_value=MagicMock(num_layers=2))
    monkeypatch.setattr(gpu_mod, "create_cache_context", create)
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    module = _bare_gpu_module()
    module.register_kv_cache(1, MagicMock(), "model", 1, MagicMock(), MagicMock(), [])
    module._cache_contexts[1].last_seen = 0.0

    module.register_kv_cache(1, MagicMock(), "model", 1, MagicMock(), MagicMock(), [])

    assert create.call_count == 1  # not rebuilt
    assert module._cache_contexts[1].last_seen > 0.0  # refreshed
    assert module._cache_contexts[1].has_liveness_signal is False


def test_gpu_touch_latches_get_does_not() -> None:
    """touch_instance marks ping-proven; get_and_touch_context_entry only refreshes."""
    module = _bare_gpu_module()
    module._cache_contexts[1] = ContextEntry(MagicMock(), "m", 1, last_seen=0.0)

    module.get_and_touch_context_entry(1)
    assert module._cache_contexts[1].last_seen > 0.0
    assert module._cache_contexts[1].has_liveness_signal is False

    module.touch_instance(1)
    assert module._cache_contexts[1].has_liveness_signal is True
    module.touch_instance(999)  # absent -> no error


def test_gpu_reap_two_tier_windows() -> None:
    """Ping-proven entries reap at the timeout; never-pinged ones survive
    until the larger registration grace."""
    module = _bare_gpu_module()
    old = time.monotonic() - 1000.0
    module._cache_contexts[1] = ContextEntry(MagicMock(), "m", 1, old, True)
    module._cache_contexts[2] = ContextEntry(MagicMock(), "m", 1, old, False)
    module._cache_contexts[3] = ContextEntry(
        MagicMock(), "m", 1, time.monotonic(), True
    )

    reaped = module.reap_stale_instances(120.0, 3600.0)

    assert reaped == [1]  # only the ping-proven stale entry
    assert module.tracked_instance_count() == 2
    cast(
        MagicMock, module._ctx.layout_desc_registry.unregister
    ).assert_called_once_with("m", 1)


def test_gpu_unregister_cleans_up() -> None:
    """unregister_kv_cache pops and releases; missing id is a no-op."""
    module = _bare_gpu_module()
    module._cache_contexts[1] = ContextEntry(MagicMock(), "m", 1, time.monotonic())

    module.unregister_kv_cache(1)
    assert module.tracked_instance_count() == 0
    cast(
        MagicMock, module._ctx.layout_desc_registry.unregister
    ).assert_called_once_with("m", 1)

    module.unregister_kv_cache(1)  # already gone -> no exception


def test_non_gpu_reap_pops_strategy_as_pair() -> None:
    """Reaping a non-GPU entry pops its strategy in the same scan, keeping
    'strategy present iff entry present'."""
    module = _bare_non_gpu_module()
    old = time.monotonic() - 1000.0
    module._engine_driven_contexts[1] = non_gpu_mod.EngineDrivenContextEntry(
        MagicMock(), "m", 1, old, True
    )
    module._strategies[1] = MagicMock()
    module._engine_driven_contexts[2] = non_gpu_mod.EngineDrivenContextEntry(
        MagicMock(), "m", 1, old, False
    )
    module._strategies[2] = MagicMock()

    reaped = module.reap_stale_instances(120.0, 3600.0)

    assert reaped == [1]
    assert 1 not in module._strategies  # strategy popped with the entry
    assert 2 in module._strategies  # never-pinged survives on grace


def test_non_gpu_resolve_for_transfer_refreshes_and_raises() -> None:
    """_resolve_for_transfer returns (entry, strategy) and refreshes
    last_seen; an unknown id raises ValueError."""
    module = _bare_non_gpu_module()
    module._engine_driven_contexts[1] = non_gpu_mod.EngineDrivenContextEntry(
        MagicMock(), "m", 1, 0.0, False
    )
    strategy = MagicMock()
    module._strategies[1] = strategy

    entry, resolved = module._resolve_for_transfer(1)
    assert resolved is strategy
    assert entry.last_seen > 0.0
    assert entry.has_liveness_signal is False  # traffic does not latch

    with pytest.raises(ValueError, match="not registered"):
        module._resolve_for_transfer(999)


class _FakeTarget:
    """Liveness target double recording touches/drops and scripted reaps."""

    def __init__(self) -> None:
        self.touched: list[int] = []
        self.to_reap: list[int] = []
        self.dropped: list[int] = []
        self.count = 0

    def touch_instance(self, instance_id: int) -> None:
        self.touched.append(instance_id)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        reaped = self.to_reap[:]
        self.to_reap.clear()
        return reaped

    def tracked_instance_count(self) -> int:
        return self.count

    def drop_instance_state(self, instance_id: int) -> None:
        self.dropped.append(instance_id)


@pytest.fixture(autouse=True)
def _reset_periodic_registry():
    """Keep the reaper out of the global registry across tests."""
    PeriodicThreadRegistry.reset()
    yield
    PeriodicThreadRegistry.reset()


def test_management_ping_touches_targets() -> None:
    """ping refreshes every target for a real id; None is ignored."""
    target = _FakeTarget()
    mgmt = ManagementModule(MagicMock(), liveness_targets=[target])

    assert mgmt.ping(42) is True
    assert mgmt.ping(None) is True
    assert target.touched == [42]


def test_management_reaper_reaps_and_drops() -> None:
    """The reaper scans targets and calls drop_instance_state for reaped ids."""
    target = _FakeTarget()
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=[target],
        worker_reap_timeout_seconds=0.4,
        worker_registration_grace_seconds=0.8,
    )
    try:
        target.to_reap = [7]
        deadline = time.monotonic() + 2.0
        while target.dropped != [7] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert target.dropped == [7]
    finally:
        mgmt.close()


def test_management_reaper_disabled_when_timeout_zero() -> None:
    """timeout == 0 starts no reaper thread."""
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=[_FakeTarget()],
        worker_reap_timeout_seconds=0.0,
    )
    assert mgmt._reaper is None
    assert mgmt.ping(1) is True


def test_management_report_status_summarizes_liveness() -> None:
    """report_status reports a worker_liveness summary when targets exist."""
    target = _FakeTarget()
    target.count = 3
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=[target],
        worker_reap_timeout_seconds=120.0,
        worker_registration_grace_seconds=3600.0,
    )
    try:
        status = mgmt.report_status()["worker_liveness"]
        assert status["enabled"] is True
        assert status["tracked_instances"] == 3
        assert status["reap_timeout_seconds"] == 120.0
    finally:
        mgmt.close()

    assert ManagementModule(MagicMock()).report_status() == {}


def test_blend_drop_instance_state_drops_rope_state() -> None:
    """drop_instance_state pops the reaped instance's CB rope state.

    The GPU context is no longer mirrored in BlendV3Module (reaping the GPU
    entry frees it directly), so only the rope state is dropped here.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.blend_v3 import BlendV3Module

    module = BlendV3Module.__new__(BlendV3Module)
    module._cb_rope_state = {5: MagicMock()}

    module.drop_instance_state(5)

    assert 5 not in module._cb_rope_state
    module.drop_instance_state(999)  # nothing held -> no error


def test_config_rejects_bad_reap_timeouts() -> None:
    """Validation rejects sub-floor reap timeouts and undersized grace."""
    with pytest.raises(ValueError, match="reap timeout"):
        MPServerConfig(worker_reap_timeout_seconds=10.0)
    with pytest.raises(ValueError, match="registration grace"):
        MPServerConfig(
            worker_reap_timeout_seconds=120.0,
            worker_registration_grace_seconds=60.0,
        )


def test_config_accepts_disabled_and_defaults() -> None:
    """0 disables reaping; defaults satisfy the grace >= timeout invariant."""
    MPServerConfig(
        worker_reap_timeout_seconds=0.0, worker_registration_grace_seconds=0.0
    )
    default = MPServerConfig()
    assert default.worker_reap_timeout_seconds == 120.0
    assert default.worker_registration_grace_seconds == 3600.0
