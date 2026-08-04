# SPDX-License-Identifier: Apache-2.0
"""Tests for NixlStorageBackend shared-pool (CPU mode) constructor-based init.

Verifies that local_cpu_backend is wired up at construction time (no post_init),
get_allocator_backend() returns the correct backend, and error cases are caught
early. No real NIXL hardware needed — agent constructors are stubbed per test,
and when the nixl package is not installed it is shadowed at sys.modules level
only for the duration of this module's import (see _install_nixl_mock_if_absent).
"""

# Standard
from unittest.mock import MagicMock
import asyncio
import sys
import types

# Third Party
import pytest
import torch

# ---------------------------------------------------------------------------
# Mock the nixl package before any import of nixl_storage_backend
# ---------------------------------------------------------------------------


def _install_nixl_mock_if_absent() -> list:
    """Install a fake ``nixl`` package so this module can import
    ``nixl_storage_backend`` and exercise its construction paths without a real
    NIXL build. Returns the list of ``sys.modules`` keys it inserted.

    The fake is removed again immediately after the imports below — it must NOT
    outlive this module's import. Other nixl test modules gate on
    ``pytest.importorskip("nixl")`` and import ``nixl`` at module top; a fake
    left in ``sys.modules`` during collection would make those modules run
    against the mock instead of skipping (or, with real nixl present, shadow it)
    and fail. Removing it at import time — rather than in a teardown — closes
    that window, because pytest imports each module to completion during
    collection before collecting the next one (a teardown runs far too late).

    When real nixl is installed we install nothing and use it as-is; the
    per-test agent ``__init__`` stubs keep the tests off any real transport.
    """
    try:
        # Third Party
        import nixl  # noqa: F401
        import nixl._api  # noqa: F401

        return []
    except ImportError:
        pass

    nixlBind_mock = MagicMock()
    nixlBind_mock.nixlRegDList = object
    nixlBind_mock.nixlXferDList = object
    nixlBind_mock.nixlBackendError = Exception

    sync_t_mock = MagicMock()
    sync_t_mock.NIXL_THREAD_SYNC_STRICT = "NIXL_THREAD_SYNC_STRICT"

    api_mock = types.ModuleType("nixl._api")
    api_mock.nixl_agent = MagicMock  # type: ignore[attr-defined]
    api_mock.nixl_agent_config = MagicMock  # type: ignore[attr-defined]
    api_mock.nixl_prepped_dlist_handle = MagicMock  # type: ignore[attr-defined]
    api_mock.nixl_xfer_handle = MagicMock  # type: ignore[attr-defined]
    api_mock.nixlBind = nixlBind_mock  # type: ignore[attr-defined]
    api_mock.nixl_thread_sync_t = sync_t_mock  # type: ignore[attr-defined]

    nixl_mock = types.ModuleType("nixl")
    nixl_mock._api = api_mock  # type: ignore[attr-defined]

    inserted = []
    for name, mod in (("nixl", nixl_mock), ("nixl._api", api_mock)):
        if name not in sys.modules:
            sys.modules[name] = mod
            inserted.append(name)
    return inserted


_NIXL_MOCK_KEYS = _install_nixl_mock_if_absent()


# First Party
from lmcache.utils import CacheEngineKey  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (  # noqa: E402
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from lmcache.v1.storage_backend import CreateStorageBackends  # noqa: E402
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend  # noqa: E402
import lmcache.v1.memory_management as memory_management_module  # noqa: E402
import lmcache.v1.storage_backend.nixl_storage_backend as nixl_module  # noqa: E402

# nixl_module has now captured whatever it needs from the (possibly fake) nixl
# package. Drop the fake from sys.modules so it cannot leak into other test
# modules' pytest.importorskip("nixl") during collection. No-op when real nixl
# is installed (nothing was inserted).
for _name in _NIXL_MOCK_KEYS:
    sys.modules.pop(_name, None)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _nixl_cpu_config(pool_size: int = 0) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="test_nixl_shared",
    )
    # In CPU mode nixl_buffer_size is rejected by config validation; the pool
    # is sized by max_local_cpu_size via LocalCPUBackend.
    config.nixl_buffer_device = "cpu"
    config.save_unfull_chunk = False
    config.extra_config = {
        "enable_nixl_storage": True,
        "nixl_backend": "OBJ",
        "nixl_pool_size": pool_size,
        "nixl_backend_params": {},
        "nixl_presence_cache": False,
        "nixl_async_put": False,
        "use_direct_io": False,
        "nixl_path": None,
        "nixl_enable_prog_thread": True,
    }
    return config


def _nixl_gpu_config(pool_size: int = 0) -> LMCacheEngineConfig:
    config = _nixl_cpu_config(pool_size)
    config.nixl_buffer_device = "cuda"
    config.nixl_buffer_size = 1024 * 1024
    return config


def _make_paged_allocator(metadata: LMCacheMetadata) -> PagedTensorMemoryAllocator:
    shapes = metadata.get_shapes()
    dtypes = metadata.get_dtypes()
    chunk_bytes = sum(
        s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True)
    )
    buffer = torch.zeros(chunk_bytes * 4, dtype=torch.uint8)
    return PagedTensorMemoryAllocator(
        buffer,
        [torch.Size(metadata.kv_shape)],
        [metadata.kv_dtype],
        MemoryFormat.KV_2LTD,
    )


def _make_local_cpu_paged(
    monkeypatch, metadata: LMCacheMetadata, local_cpu: bool = True
) -> LocalCPUBackend:
    """LocalCPUBackend with memory_allocator = MixedMemoryAllocator(use_paging=True).

    Sized to exactly 4 chunks so tests that need to drive the pool to
    saturation know the page count without inspecting allocator internals.
    """
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=local_cpu, lmcache_instance_id="test_paged"
    )
    shapes = metadata.get_shapes()
    dtypes = metadata.get_dtypes()
    chunk_bytes = sum(
        s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True)
    )
    aligned = chunk_bytes * 4
    config.max_local_cpu_size = aligned / (1024**3)
    config.nixl_buffer_device = "cpu"
    config.extra_config = {"enable_nixl_storage": True}
    config.save_unfull_chunk = False
    real_buf = torch.zeros(aligned, dtype=torch.uint8)
    monkeypatch.setattr(
        memory_management_module,
        "_allocate_cpu_memory",
        lambda size, *a, **kw: real_buf,
    )
    # real_buf is a plain (pageable) tensor, not cudaHostAlloc'd. The real
    # _free_cpu_memory would call cudaFreeHost on it on a CUDA build and raise
    # "cudaFreeHost failed" — so neutralize free symmetrically with alloc; the
    # tensor is reclaimed by normal Python GC.
    monkeypatch.setattr(
        memory_management_module, "_free_cpu_memory", lambda *a, **kw: None
    )
    return LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")


def _make_local_cpu_flat(metadata: LMCacheMetadata) -> LocalCPUBackend:
    """LocalCPUBackend memory_allocator = MixedMemoryAllocator(use_paging=False)."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, lmcache_instance_id="test_flat"
    )
    config.max_local_cpu_size = 0.01
    return LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")


def _stub_dynamic_agent(monkeypatch) -> None:
    monkeypatch.setattr(
        nixl_module.NixlDynamicStorageAgent, "__init__", lambda self, *a, **kw: None
    )
    monkeypatch.setattr(nixl_module.NixlDynamicStorageAgent, "close", lambda self: None)
    monkeypatch.setattr(
        nixl_module.NixlDynamicStorageAgent, "mem_type", "OBJ", raising=False
    )


def _stub_static_agent(monkeypatch) -> None:
    monkeypatch.setattr(
        nixl_module.NixlStaticStorageAgent, "__init__", lambda self, *a, **kw: None
    )
    monkeypatch.setattr(nixl_module.NixlStaticStorageAgent, "close", lambda self: None)


def _stub_gpu_allocator(monkeypatch, allocator) -> None:
    monkeypatch.setattr(
        nixl_module.NixlStorageBackend,
        "initialize_allocator",
        lambda self, config, metadata: allocator,
    )


def _build_dynamic(monkeypatch, metadata, config, local_cpu_backend=None):
    _stub_dynamic_agent(monkeypatch)
    loop = asyncio.new_event_loop()
    nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(
        config, metadata
    )
    backend = nixl_module.NixlDynamicStorageBackend(
        nixl_config, config, metadata, loop, local_cpu_backend=local_cpu_backend
    )
    return backend, loop


def _build_static(monkeypatch, metadata, config, local_cpu_backend=None):
    _stub_static_agent(monkeypatch)
    loop = asyncio.new_event_loop()
    nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(
        config, metadata
    )
    backend = nixl_module.NixlStaticStorageBackend(
        nixl_config, config, metadata, loop, local_cpu_backend=local_cpu_backend
    )
    return backend, loop


def _make_key(idx: int) -> CacheEngineKey:
    """Unique CacheEngineKey for tests."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=idx,
        dtype=torch.bfloat16,
    )


def _stub_dynamic_agent_runtime(backend) -> None:
    """Stub the NixlDynamicStorageAgent runtime methods invoked by storage_to_mem.

    The constructor was already stubbed by _stub_dynamic_agent; this fills in
    the transport surface so storage_to_mem can run end-to-end without real NIXL.
    """
    backend.agent.mem_type = "OBJ"
    backend.agent.create_batched_storage_handler = MagicMock(
        return_value=(MagicMock(), MagicMock())
    )
    backend.agent.get_storage_to_mem_handle = MagicMock(return_value=MagicMock())
    backend.agent.post_blocking = MagicMock()
    backend.agent.release_handle = MagicMock()
    backend.agent.release_storage_handler = MagicMock()


# ---------------------------------------------------------------------------
# Tests: Dynamic CPU mode
# ---------------------------------------------------------------------------


class TestDynamicCpuMode:
    def test_cpu_mode_raises_without_local_cpu(self, monkeypatch):
        """Constructor raises RuntimeError if local_cpu_backend is None in CPU mode."""
        metadata = _make_metadata()
        _stub_dynamic_agent(monkeypatch)
        loop = asyncio.new_event_loop()
        try:
            nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(
                _nixl_cpu_config(), metadata
            )
            with pytest.raises(RuntimeError, match="max_local_cpu_size"):
                nixl_module.NixlDynamicStorageBackend(
                    nixl_config,
                    _nixl_cpu_config(),
                    metadata,
                    loop,
                    local_cpu_backend=None,
                )
        finally:
            loop.close()

    def test_cpu_mode_raises_if_wrong_allocator_type(self, monkeypatch):
        """RuntimeError when LocalCPUBackend uses non-paged MixedMemoryAllocator."""
        metadata = _make_metadata()
        local_cpu_flat = _make_local_cpu_flat(metadata)
        _stub_dynamic_agent(monkeypatch)
        loop = asyncio.new_event_loop()
        try:
            nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(
                _nixl_cpu_config(), metadata
            )
            with pytest.raises(
                RuntimeError, match="MixedMemoryAllocator\\(use_paging=True\\)"
            ):
                nixl_module.NixlDynamicStorageBackend(
                    nixl_config,
                    _nixl_cpu_config(),
                    metadata,
                    loop,
                    local_cpu_backend=local_cpu_flat,
                )
        finally:
            loop.close()

    def test_get_allocator_backend_returns_local_cpu(self, monkeypatch):
        """get_allocator_backend() returns local_cpu_backend in CPU mode."""
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        backend, loop = _build_dynamic(
            monkeypatch, metadata, _nixl_cpu_config(), local_cpu_backend=local_cpu
        )
        try:
            assert backend.get_allocator_backend() is local_cpu
        finally:
            loop.close()
            local_cpu.memory_allocator.close()


# ---------------------------------------------------------------------------
# Tests: Dynamic GPU mode
# ---------------------------------------------------------------------------


class TestDynamicGpuMode:
    def test_get_allocator_backend_returns_self(self, monkeypatch):
        """In GPU mode, get_allocator_backend() returns self."""
        metadata = _make_metadata()
        fake_alloc = _make_paged_allocator(metadata)
        _stub_gpu_allocator(monkeypatch, fake_alloc)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
        monkeypatch.setattr(torch.cuda, "set_device", lambda x: None, raising=False)

        backend, loop = _build_dynamic(monkeypatch, metadata, _nixl_gpu_config())
        try:
            assert backend.get_allocator_backend() is backend
        finally:
            loop.close()
            fake_alloc.close()


# ---------------------------------------------------------------------------
# Tests: Static CPU mode
# ---------------------------------------------------------------------------


class TestStaticCpuMode:
    def test_cpu_mode_raises_if_wrong_allocator_type(self, monkeypatch):
        """Static backend: RuntimeError when LocalCPUBackend uses flat allocator."""
        metadata = _make_metadata()
        local_cpu_flat = _make_local_cpu_flat(metadata)
        _stub_static_agent(monkeypatch)
        loop = asyncio.new_event_loop()
        try:
            nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(
                _nixl_cpu_config(pool_size=4), metadata
            )
            with pytest.raises(
                RuntimeError, match="MixedMemoryAllocator\\(use_paging=True\\)"
            ):
                nixl_module.NixlStaticStorageBackend(
                    nixl_config,
                    _nixl_cpu_config(pool_size=4),
                    metadata,
                    loop,
                    local_cpu_backend=local_cpu_flat,
                )
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Tests: cross-pool ownership bug fix (red->green demonstration)
# ---------------------------------------------------------------------------


class TestCrossPoolOwnershipFix:
    """Red->green test for the cross-pool ownership bug.

    Pre-fix: _allocate_for_read called self.memory_allocator.allocate()
    directly. Once NIXL-hit promotions filled LocalCPUBackend's hot cache
    with buffers from the shared pool, allocate() returned None and
    batched_get_blocking returned [None].

    Post-fix: _allocate_for_read routes through
    LocalCPUBackend.allocate(eviction=True), so eviction fires on the
    shared pool and frees a page from hot_cache to satisfy the read.
    """

    def test_get_succeeds_after_pool_saturation_via_eviction(self, monkeypatch):
        """After N promotions saturate the shared pool, the (N+1)th
        batched_get_blocking still returns a valid MemoryObj (eviction fires
        via the shared allocator)."""
        metadata = _make_metadata()
        # The fixture sizes the pool to exactly 4 chunks; drive exactly that
        # many promotions to saturate it without peeking at allocator state.
        num_pages = 4
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        backend, loop = _build_dynamic(
            monkeypatch, metadata, _nixl_cpu_config(), local_cpu_backend=local_cpu
        )

        try:
            backend.init_chunk_meta(metadata)
            _stub_dynamic_agent_runtime(backend)

            for i in range(num_pages):
                key = _make_key(i)
                objs = backend.batched_get_blocking([key])
                assert objs[0] is not None, (
                    f"batched_get_blocking returned None at promotion "
                    f"{i}/{num_pages}; pool exhausted earlier than expected"
                )
                # Promote into hot_cache and drop the caller's ref so hot_cache
                # holds the only reference to the page.
                local_cpu.batched_submit_put_task([key], objs)
                objs[0].ref_count_down()

            # Verify the pool is saturated via the public allocator API: a
            # no-eviction allocation must fail when every page is pinned in
            # hot_cache.
            shapes = metadata.get_shapes()
            dtypes = metadata.get_dtypes()
            sat_obj = local_cpu.allocate(
                shapes, dtypes, MemoryFormat.KV_2LTD, eviction=False
            )
            assert sat_obj is None, (
                "Test fixture: shared pool should be saturated after "
                f"{num_pages} promotions"
            )

            # The (N+1)th read must still succeed — _allocate_for_read routes
            # through LocalCPUBackend.allocate(eviction=True), which evicts
            # one hot_cache entry to free a page.
            extra_key = _make_key(num_pages)
            objs = backend.batched_get_blocking([extra_key])
            assert objs[0] is not None, (
                "Shared-pool eviction did not fire: pre-fix, _allocate_for_read "
                "called the inner allocator directly with no eviction and "
                "returned None."
            )
            objs[0].ref_count_down()
        finally:
            loop.close()
            local_cpu.memory_allocator.close()


# ---------------------------------------------------------------------------
# Tests: shared-pool contract (close ownership + staging-only mode)
# ---------------------------------------------------------------------------


class TestSharedPoolContract:
    """Behavioral contracts for the CPU-mode shared pool."""

    def test_close_does_not_close_borrowed_allocator(self, monkeypatch):
        """NixlStorageBackend.close() must not close the LocalCPUBackend
        allocator it borrowed — that allocator is owned by LocalCPUBackend."""
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        backend, loop = _build_dynamic(
            monkeypatch, metadata, _nixl_cpu_config(), local_cpu_backend=local_cpu
        )

        close_calls: list[None] = []
        original_close = local_cpu.memory_allocator.close

        def spy() -> None:
            close_calls.append(None)
            original_close()

        monkeypatch.setattr(local_cpu.memory_allocator, "close", spy)

        try:
            backend.close()
            assert close_calls == [], (
                "NIXL.close() must not invoke close() on the borrowed allocator"
            )
            # And the allocator is still usable after NIXL has shut down.
            shapes = metadata.get_shapes()
            dtypes = metadata.get_dtypes()
            obj = local_cpu.allocate(shapes, dtypes, MemoryFormat.KV_2LTD)
            assert obj is not None, (
                "Borrowed allocator is no longer usable after NIXL close()"
            )
            obj.ref_count_down()
        finally:
            loop.close()
            local_cpu.memory_allocator.close()

    def test_local_cpu_false_does_not_promote_to_hot_cache(self, monkeypatch):
        """With local_cpu=False the LocalCPUBackend is a NIXL staging buffer
        only: batched_submit_put_task is a no-op, so a subsequent
        get_blocking returns None. Documents the contract stated in the
        NIXL docs and exercised by the CreateStorageBackends path that
        force-creates LocalCPUBackend in CPU mode."""
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata, local_cpu=False)

        try:
            shapes = metadata.get_shapes()
            dtypes = metadata.get_dtypes()
            obj = local_cpu.allocate(shapes, dtypes, MemoryFormat.KV_2LTD)
            assert obj is not None, "Staging-only LocalCPUBackend must still allocate"
            key = _make_key(0)
            local_cpu.batched_submit_put_task([key], [obj])
            assert local_cpu.get_blocking(key) is None, (
                "local_cpu=False must suppress hot-cache promotion"
            )
            obj.ref_count_down()
        finally:
            local_cpu.memory_allocator.close()


class TestAllocatorMethodsRouteThroughLocalCpu:
    """In CPU mode the AllocatorBackendInterface methods on NixlStorageBackend
    must delegate to LocalCPUBackend's public API, not call the inner
    allocator directly. Matches P2P's pattern (allocations via
    local_cpu_backend.allocate, allocator only peeked for buffer info)."""

    def test_allocate_routes_through_local_cpu_backend(self, monkeypatch):
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        backend, loop = _build_dynamic(
            monkeypatch, metadata, _nixl_cpu_config(), local_cpu_backend=local_cpu
        )

        calls: list[dict] = []
        original_allocate = local_cpu.allocate

        def spy(shapes, dtypes, fmt=None, eviction=True, busy_loop=True):
            calls.append(
                {"shapes": shapes, "eviction": eviction, "busy_loop": busy_loop}
            )
            return original_allocate(
                shapes, dtypes, fmt, eviction=eviction, busy_loop=busy_loop
            )

        monkeypatch.setattr(local_cpu, "allocate", spy)

        try:
            shapes = metadata.get_shapes()
            dtypes = metadata.get_dtypes()
            obj = backend.allocate(shapes, dtypes, MemoryFormat.KV_2LTD)
            assert obj is not None
            assert len(calls) == 1, (
                "NixlStorageBackend.allocate must delegate to LocalCPUBackend"
            )
            assert calls[0]["busy_loop"] is False, (
                "Delegated allocate must force busy_loop=False (NIXL semantics)"
            )
            obj.ref_count_down()
        finally:
            backend.close()
            loop.close()
            local_cpu.memory_allocator.close()

    def test_batched_allocate_routes_through_local_cpu_backend(self, monkeypatch):
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        backend, loop = _build_dynamic(
            monkeypatch, metadata, _nixl_cpu_config(), local_cpu_backend=local_cpu
        )

        calls: list[dict] = []
        original_batched = local_cpu.batched_allocate

        def spy(shapes, dtypes, batch_size, fmt=None, eviction=True, busy_loop=True):
            calls.append({"batch_size": batch_size, "busy_loop": busy_loop})
            return original_batched(
                shapes,
                dtypes,
                batch_size,
                fmt,
                eviction=eviction,
                busy_loop=busy_loop,
            )

        monkeypatch.setattr(local_cpu, "batched_allocate", spy)

        try:
            shapes = metadata.get_shapes()
            dtypes = metadata.get_dtypes()
            objs = backend.batched_allocate(shapes, dtypes, 2, MemoryFormat.KV_2LTD)
            assert objs is not None
            assert len(calls) == 1, (
                "NixlStorageBackend.batched_allocate must delegate to LocalCPUBackend"
            )
            assert calls[0]["batch_size"] == 2
            assert calls[0]["busy_loop"] is False
            for obj in objs:
                obj.ref_count_down()
        finally:
            backend.close()
            loop.close()
            local_cpu.memory_allocator.close()

    def test_get_memory_allocator_returns_local_cpu_allocator(self, monkeypatch):
        """storage_manager.memcheck() iterates AllocatorBackendInterface
        backends and calls get_memory_allocator().memcheck(). In CPU mode the
        NIXL backend shares LocalCPUBackend's pool — returning the same
        allocator avoids double-counting the pool in memcheck."""
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        backend, loop = _build_dynamic(
            monkeypatch, metadata, _nixl_cpu_config(), local_cpu_backend=local_cpu
        )

        try:
            assert backend.get_memory_allocator() is local_cpu.get_memory_allocator()
        finally:
            backend.close()
            loop.close()
            local_cpu.memory_allocator.close()


def _scheduler_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
        role="scheduler",
    )


class TestSchedulerRoleRejection:
    """The scheduler role never creates a LocalCPUBackend, so NIXL CPU mode —
    which shares that backend's pool — cannot run there. CreateStorageBackends
    rejects the combo with a clear ValueError instead of letting it crash deep
    in the NIXL constructor. See plans/pr-nixl-cpu-shared-pool-review."""

    def test_scheduler_role_nixl_cpu_rejected(self):
        config = _nixl_cpu_config()
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(ValueError, match="scheduler"):
                CreateStorageBackends(config, _scheduler_metadata(), loop)
        finally:
            loop.close()

    def test_scheduler_role_without_nixl_ok(self):
        # The guard is specific to NIXL CPU mode: a scheduler with no NIXL
        # storage must still build (no LocalCPUBackend, no rejection).
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            lmcache_instance_id="test_sched_no_nixl",
        )
        loop = asyncio.new_event_loop()
        try:
            backends = CreateStorageBackends(config, _scheduler_metadata(), loop)
            assert "LocalCPUBackend" not in backends
        finally:
            loop.close()
