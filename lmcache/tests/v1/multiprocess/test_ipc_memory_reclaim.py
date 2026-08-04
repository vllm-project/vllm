# SPDX-License-Identifier: Apache-2.0
"""CUDA-IPC memory reclaim on instance release (LMCache#4014).

The server imports each client's KV pool over CUDA IPC; when an instance is
released (unregister / reaper / close) those imported segments are only
returned to the driver by an ``empty_cache()`` + ``ipc_collect()`` pass run
AFTER every reference to the released entry is gone.

All tests drive the module through its public surface: the real constructor,
``register_kv_cache`` (with the module-level context factory stubbed),
``unregister_kv_cache`` / ``reap_stale_instances`` / ``close``, and
``context_entries_snapshot`` for reads. The stubbed boundaries are external
by nature: the GPU context factory, event IPC backend lookup, and the device
module (``torch_dev``).
"""

# Standard
# Standard Library
from types import SimpleNamespace
from unittest.mock import MagicMock
import time
import weakref

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as gpu_mod


class _FakeTorchDev:
    """Records the reclaim-call sequence; optionally omits ipc_collect."""

    empty_cache: MagicMock
    ipc_collect: MagicMock

    def __init__(self, with_ipc_collect: bool = True):
        self.calls: list[str] = []
        self.empty_cache = MagicMock(
            side_effect=lambda: self.calls.append("empty_cache")
        )
        if with_ipc_collect:
            self.ipc_collect = MagicMock(
                side_effect=lambda: self.calls.append("ipc_collect")
            )


def _module(monkeypatch) -> LMCacheDrivenTransferModule:
    """Construct the module through the real __init__ with stubbed deps."""
    monkeypatch.setattr(gpu_mod, "DeviceHostFuncDispatcher", MagicMock())
    return LMCacheDrivenTransferModule(MagicMock(name="ctx"))


def _register(
    module: LMCacheDrivenTransferModule,
    monkeypatch,
    instance_id: int,
    model: str = "m",
    age_s: float = 0.0,
) -> MagicMock:
    """Register an instance via the public API; return its cache context.

    ``age_s`` back-dates the registration (by stubbing the clock for the
    duration of the call) so reaper tests can create already-stale entries
    without touching module internals.

    Returns:
        The MagicMock standing in for the created cache context.
    """
    cache_context = MagicMock(name=f"cache_context-{instance_id}")
    cache_context.num_layers = 1
    event_backend = MagicMock(name=f"event_backend-{instance_id}")
    monkeypatch.setattr(gpu_mod, "create_cache_context", lambda *a, **kw: cache_context)
    monkeypatch.setattr(
        gpu_mod,
        "get_event_ipc_backend",
        lambda device: event_backend,
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    real_monotonic = time.monotonic
    if age_s:
        monkeypatch.setattr(gpu_mod.time, "monotonic", lambda: real_monotonic() - age_s)
    try:
        module.register_kv_cache(
            instance_id,
            kv_caches=MagicMock(name="kv_caches"),
            model_name=model,
            world_size=1,
            engine_type=MagicMock(name="engine_type"),
            layout_hints=MagicMock(name="layout_hints"),
            engine_group_infos=[],
        )
    finally:
        if age_s:
            monkeypatch.setattr(gpu_mod.time, "monotonic", real_monotonic)
    return cache_context


def test_unregister_reclaims_ipc_memory(monkeypatch) -> None:
    """Explicit unregister closes the context AND runs empty_cache +
    ipc_collect (in that order)."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    ctx = _register(module, monkeypatch, 7)

    module.unregister_kv_cache(7)

    ctx.close.assert_called_once()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert module.context_entries_snapshot() == {}


def test_unregister_unknown_instance_does_not_reclaim(monkeypatch) -> None:
    """The warn path (already-reaped / never-registered id) must not touch
    the allocator — reclaim is tied to an actual release."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)

    module.unregister_kv_cache(404)

    assert dev.calls == []


def test_unregister_entry_refs_dead_before_ipc_collect(monkeypatch) -> None:
    """THE load-bearing ordering: ipc_collect only frees segments whose
    tensors are unreferenced, so the entry must be garbage by the time it
    fires. Verified with a weakref probed from inside the fake collector."""
    module = _module(monkeypatch)
    _register(module, monkeypatch, 1)
    ref = weakref.ref(module.context_entries_snapshot()[1])

    seen: dict = {}
    dev = SimpleNamespace(
        empty_cache=lambda: None,
        ipc_collect=lambda: seen.setdefault("entry_alive", ref() is not None),
    )
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)

    module.unregister_kv_cache(1)

    assert seen == {"entry_alive": False}


def test_reaper_reclaims_once_per_batch(monkeypatch) -> None:
    """Reaping N stale instances closes each context but batches the
    allocator reclaim into ONE empty_cache + ipc_collect."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    ctx_a = _register(module, monkeypatch, 1, model="a", age_s=1000.0)
    ctx_b = _register(module, monkeypatch, 2, model="b", age_s=1000.0)
    ctx_fresh = _register(module, monkeypatch, 3, model="c")

    reaped = module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)

    assert sorted(reaped) == [1, 2]
    ctx_a.close.assert_called_once()
    ctx_b.close.assert_called_once()
    ctx_fresh.close.assert_not_called()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert list(module.context_entries_snapshot()) == [3]


def test_reaper_noop_scan_does_not_reclaim(monkeypatch) -> None:
    """A scan that reaps nothing must not thrash the allocator."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    _register(module, monkeypatch, 1)

    reaped = module.reap_stale_instances(
        reap_timeout_s=3600.0, registration_grace_s=3600.0
    )

    assert reaped == []
    assert dev.calls == []


def test_reaper_entry_refs_dead_before_ipc_collect(monkeypatch) -> None:
    """Same ref-lifetime invariant on the reaper path."""
    module = _module(monkeypatch)
    _register(module, monkeypatch, 1, age_s=1000.0)
    ref = weakref.ref(module.context_entries_snapshot()[1])

    seen: dict = {}
    dev = SimpleNamespace(
        empty_cache=lambda: None,
        ipc_collect=lambda: seen.setdefault("entry_alive", ref() is not None),
    )
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)

    module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)

    assert seen == {"entry_alive": False}


def test_reclaim_degrades_without_ipc_collect(monkeypatch) -> None:
    """Device modules without ipc_collect (xpu / musa) must not raise —
    empty_cache still runs, the collect step is skipped."""
    dev = _FakeTorchDev(with_ipc_collect=False)
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    _register(module, monkeypatch, 9)

    module.unregister_kv_cache(9)

    assert dev.calls == ["empty_cache"]


def test_close_releases_all_and_reclaims_once(monkeypatch) -> None:
    """Server close() releases every remaining context and reclaims once."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    c1 = _register(module, monkeypatch, 1, model="a")
    c2 = _register(module, monkeypatch, 2, model="b")

    module.close()

    c1.close.assert_called_once()
    c2.close.assert_called_once()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert module.context_entries_snapshot() == {}


def test_close_with_empty_registry_does_not_reclaim(monkeypatch) -> None:
    """close() on a server that never had clients skips the allocator."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)

    module.close()

    assert dev.calls == []
