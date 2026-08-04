# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock, patch
import asyncio
import os
import struct
import sys
import tempfile
import threading
import time
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    _DEFAULT_META_MAGIC,
    _DEFAULT_META_VERSION,
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    RawBlockCoreConfig,
    RawBlockKeySpec,
    RawBlockPutManyResult,
)


def _has_ext() -> bool:
    try:
        # Third Party
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except Exception:
        return False


class _FakeRawBlockDevice:
    def __init__(self, path: str, *, size_bytes: int, **kwargs):
        del path, kwargs
        self._data = bytearray(size_bytes)

    def size_bytes(self):
        return len(self._data)

    def pread_into(self, offset, out, payload_len, total_len=None):
        del total_len
        out[:payload_len] = self._data[offset : offset + payload_len]

    def pwrite_from_buffer(self, offset, data, payload_len=None, total_len=None):
        del total_len
        length = len(data) if payload_len is None else payload_len
        self._data[offset : offset + length] = bytes(memoryview(data)[:length])

    def close(self):
        return None


def _install_fake_raw_block_device(monkeypatch, *, size_bytes: int = 64 * 1024):
    def create_fake_device(path: str, **kwargs):
        return _FakeRawBlockDevice(path, size_bytes=size_bytes, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(RawBlockDevice=create_fake_device),
    )


def _make_raw_block_core(*, use_odirect: bool = False) -> RawBlockCore:
    return RawBlockCore(
        RawBlockCoreConfig(
            device_path="/tmp/raw-block-boundary-test",
            capacity_bytes=64 * 1024,
            block_align=4096,
            header_bytes=4096,
            slot_bytes=8192,
            use_odirect=use_odirect,
            enable_zero_copy=True,
            meta_total_bytes=16 * 1024,
            meta_magic=b"LMCIDX01",
            meta_version=1,
            meta_checkpoint_interval_sec=60,
            meta_idle_quiet_ms=100,
            meta_enable_periodic=False,
            load_checkpoint_on_init=True,
            meta_verify_on_load=True,
            io_engine="posix",
            iouring_queue_depth=256,
        ),
        key_namespace="object",
    )


@pytest.mark.parametrize("block_align", [0, -1, 3, 4095])
def test_raw_block_core_rejects_non_power_of_2_block_align(block_align: int) -> None:
    """RawBlockCore rejects invalid block_align values."""
    with pytest.raises(ValueError, match="block_align"):
        RawBlockCore(
            RawBlockCoreConfig(
                device_path="/tmp/dummy-does-not-need-to-exist",
                capacity_bytes=64 * 1024,
                block_align=block_align,
                header_bytes=4096,
                slot_bytes=8192,
                use_odirect=False,
                enable_zero_copy=True,
                meta_total_bytes=16 * 1024,
                meta_magic=b"LMCIDX01",
                meta_version=1,
                meta_checkpoint_interval_sec=60,
                meta_idle_quiet_ms=100,
                meta_enable_periodic=False,
                load_checkpoint_on_init=False,
                meta_verify_on_load=False,
                io_engine="posix",
                iouring_queue_depth=256,
            ),
            key_namespace="object",
        )


def _make_byte_obj(size: int) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.uint8,
        address=0,
        phy_size=size,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def test_raw_block_core_passes_io_engine_options_to_rust_binding(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeRawBlockDevice:
        def __init__(self, path: str, **kwargs):
            calls.append({"path": path, **kwargs})

        def size_bytes(self):
            return 2 * 1024 * 1024

        def pread_into(self, offset, out, payload_len, total_len=None):
            del offset, total_len
            out[:payload_len] = b"\x00" * payload_len

        def pwrite_from_buffer(self, offset, data, payload_len=None, total_len=None):
            del offset, data, payload_len, total_len

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(RawBlockDevice=FakeRawBlockDevice),
    )

    core = RawBlockCore(
        RawBlockCoreConfig(
            device_path="/tmp/raw-block-engine-test",
            capacity_bytes=0,
            block_align=4096,
            header_bytes=4096,
            slot_bytes=8192,
            use_odirect=False,
            enable_zero_copy=True,
            meta_total_bytes=16 * 1024,
            meta_magic=b"LMCIDX01",
            meta_version=1,
            meta_checkpoint_interval_sec=60,
            meta_idle_quiet_ms=100,
            meta_enable_periodic=False,
            load_checkpoint_on_init=True,
            meta_verify_on_load=True,
            io_engine="io_uring",
            iouring_queue_depth=512,
        ),
        key_namespace="legacy",
    )
    try:
        assert calls == [
            {
                "path": "/tmp/raw-block-engine-test",
                "writable": True,
                "use_odirect": False,
                "alignment": 4096,
                "io_engine": "io_uring",
                "use_uring_cmd": False,
                "iouring_queue_depth": 512,
            }
        ]
    finally:
        core.close()


def test_raw_block_core_non_odirect_rejects_payload_over_slot_capacity(monkeypatch):
    _install_fake_raw_block_device(monkeypatch)
    core = _make_raw_block_core(use_odirect=False)
    try:
        payload_capacity = core.slot_bytes - core.header_bytes
        exact_key = RawBlockKeySpec(encoded="exact", slot_identity=1)
        too_large_key = RawBlockKeySpec(encoded="too-large", slot_identity=2)

        exact = core.put_many([exact_key], [_make_byte_obj(payload_capacity)])
        assert exact.results == [True]
        assert core.contains_key("exact") is True

        too_large = core.put_many(
            [too_large_key],
            [_make_byte_obj(payload_capacity + 1)],
        )
        assert too_large.results == [False]
        assert core.contains_key("too-large") is False
    finally:
        core.close()


def test_raw_block_core_odirect_prepare_payload_boundaries(monkeypatch):
    _install_fake_raw_block_device(monkeypatch)
    core = _make_raw_block_core(use_odirect=True)
    try:
        payload_capacity = core.slot_bytes - core.header_bytes

        _, payload_len, total_len = core._prepare_write_payload(
            _make_byte_obj(payload_capacity)
        )
        assert payload_len == payload_capacity
        assert total_len == payload_capacity

        _, payload_len, total_len = core._prepare_write_payload(
            _make_byte_obj(payload_capacity - 1)
        )
        assert payload_len == payload_capacity - 1
        assert total_len == payload_capacity

        with pytest.raises(RuntimeError, match="slot capacity"):
            core._prepare_write_payload(_make_byte_obj(payload_capacity + 1))
    finally:
        core.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
@pytest.mark.parametrize("io_engine", ["posix", "io_uring"])
def test_raw_block_device_all_io_engines_roundtrip_on_tmp_file(io_engine):
    # Third Party
    from lmcache_rust_raw_block_io import RawBlockDevice

    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        dev_path = os.path.join(td, f"raw-block-{io_engine}.bin")
        with open(dev_path, "wb") as f:
            f.truncate(1024 * 1024)

        dev = RawBlockDevice(
            dev_path,
            writable=True,
            use_odirect=False,
            alignment=4096,
            io_engine=io_engine,
        )

        try:
            payload = bytearray(f"raw-block-{io_engine}".encode())
            out = bytearray(len(payload))
            dev.pwrite_from_buffer(4096, payload, len(payload), len(payload))
            dev.pread_into(4096, out, len(out), len(out))
            assert out == payload
        finally:
            dev.close()


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="test-loop", daemon=True)
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


def _run_batched_get_prefix_stop(
    memory_allocator,
    loop_in_thread,
    *,
    async_get: bool,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_batched_get",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = AdHocMemoryAllocator(device="cpu")
            key1 = CacheEngineKey("test_model", 1, 0, 1001, torch.bfloat16)
            key_miss = CacheEngineKey("test_model", 1, 0, 1002, torch.bfloat16)
            key3 = CacheEngineKey("test_model", 1, 0, 1003, torch.bfloat16)

            obj1 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            obj3 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj1 is not None and obj1.tensor is not None
            assert obj3 is not None and obj3.tensor is not None
            obj1.tensor.fill_(1)
            obj3.tensor.fill_(3)
            expected1 = bytes(obj1.byte_array)
            expected3 = bytes(obj3.byte_array)

            for key, obj in ((key1, obj1), (key3, obj3)):
                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

            if async_get:
                future = asyncio.run_coroutine_threadsafe(
                    backend.batched_get_non_blocking(
                        "lookup-rawblock", [key1, key_miss, key3]
                    ),
                    loop_in_thread,
                )
                async_results = future.result(timeout=10)
                assert len(async_results) == 1
                assert bytes(async_results[0].byte_array) == expected1
                async_results[0].ref_count_down()
            else:
                blocking_results = backend.batched_get_blocking([key1, key_miss, key3])
                assert len(blocking_results) == 3
                assert blocking_results[0] is not None
                assert bytes(blocking_results[0].byte_array) == expected1
                assert blocking_results[1] is None
                assert blocking_results[2] is None
                blocking_results[0].ref_count_down()

            out3 = backend.get_blocking(key3)
            assert out3 is not None
            assert bytes(out3.byte_array) == expected3
            out3.ref_count_down()
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_put_get_roundtrip(memory_allocator, loop_in_thread):
    """Test basic put/get roundtrip with RustRawBlockBackend."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 12345, torch.bfloat16)
            allocator = AdHocMemoryAllocator(device="cpu")
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(7)
            expected = bytes(obj.byte_array)

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            assert isinstance(futs[0], Future)
            futs[0].result(timeout=10)

            out = backend.get_blocking(key)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_blocking_prefix_stop(
    memory_allocator, loop_in_thread
):
    """Batched blocking get should stop at the first miss and preserve order."""
    _run_batched_get_prefix_stop(memory_allocator, loop_in_thread, async_get=False)


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_non_blocking_prefix_stop(
    memory_allocator, loop_in_thread
):
    """Async batched get should return only the successful prefix."""
    _run_batched_get_prefix_stop(memory_allocator, loop_in_thread, async_get=True)


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_pin_and_contains_are_idempotent(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_pin_idempotent",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 4242, torch.bfloat16)
            allocator = AdHocMemoryAllocator(device="cpu")
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()

            encoded_key = key.to_string()
            assert backend.contains(key, pin=True) is True
            assert backend.lock_refcount(encoded_key) == 1
            assert backend.contains(key, pin=True) is True
            assert backend.lock_refcount(encoded_key) == 1
            assert backend.pin(key) is True
            assert backend.lock_refcount(encoded_key) == 1

            assert backend.unpin(key) is True
            assert backend.lock_refcount(encoded_key) == 0
            assert backend.unpin(key) is True
            assert backend.lock_refcount(encoded_key) == 0

            barrier = threading.Barrier(8)
            pin_results: list[bool] = []
            result_lock = threading.Lock()

            def pin_concurrently() -> None:
                barrier.wait(timeout=5)
                pinned = backend.pin(key)
                with result_lock:
                    pin_results.append(pinned)

            threads = [
                threading.Thread(target=pin_concurrently, daemon=True) for _ in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            assert pin_results == [True] * 8
            assert backend.lock_refcount(encoded_key) == 1

            assert backend.unpin(key) is True
            assert backend.lock_refcount(encoded_key) == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_close_is_thread_safe(memory_allocator, loop_in_thread):
    """Concurrent close calls should not double-clean raw-block resources."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_concurrent_close",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def close_backend() -> None:
            try:
                backend.close()
            except BaseException as e:
                with errors_lock:
                    errors.append(e)

        threads = [threading.Thread(target=close_backend) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert errors == []
        backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_resets_inflight_on_rawdev_error(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_rawdev_error",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 3001, torch.bfloat16)
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(31)
            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()

            with patch.object(
                backend._core, "_rawdev", side_effect=RuntimeError("boom")
            ):
                with pytest.raises(RuntimeError, match="boom"):
                    backend.get_blocking(key)
            assert backend.inflight_io_count() == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_handles_allocator_exhaustion(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_allocator_exhaustion",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 3002, torch.bfloat16)
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(32)
            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()

            with patch.object(local_cpu, "allocate", return_value=None):
                assert backend.get_blocking(key) is None
                assert backend.batched_get_blocking([key]) == [None]
            assert backend.inflight_io_count() == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_releases_allocation_on_read_error(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_release_failed_get",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 3003, torch.bfloat16)
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(33)
            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()

            leaked_obj = local_cpu.allocate(
                torch.Size([2, 16, 8, 128]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
            assert leaked_obj is not None
            assert leaked_obj.get_ref_count() == 1

            raw_dev = MagicMock()
            raw_dev.pread_into.side_effect = OSError("read failed")
            with patch.object(local_cpu, "allocate", return_value=leaked_obj):
                with patch.object(backend._core, "_rawdev", return_value=raw_dev):
                    with pytest.raises(OSError, match="read failed"):
                        backend.get_blocking(key)

            assert leaked_obj.get_ref_count() == 0
            assert backend.inflight_io_count() == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_releases_loaded_prefix_on_read_error(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_release_prefix_get",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = AdHocMemoryAllocator(device="cpu")
            key1 = CacheEngineKey("test_model", 1, 0, 3004, torch.bfloat16)
            key2 = CacheEngineKey("test_model", 1, 0, 3005, torch.bfloat16)
            for key, fill in ((key1, 34), (key2, 35)):
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None and obj.tensor is not None
                obj.tensor.fill_(fill)
                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

            loaded_obj = local_cpu.allocate(
                torch.Size([2, 16, 8, 128]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
            failed_obj = local_cpu.allocate(
                torch.Size([2, 16, 8, 128]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
            assert loaded_obj is not None
            assert failed_obj is not None
            assert loaded_obj.get_ref_count() == 1
            assert failed_obj.get_ref_count() == 1

            raw_dev = MagicMock()
            raw_dev.pread_into.side_effect = [None, OSError("read failed")]
            with patch.object(
                local_cpu, "allocate", side_effect=[loaded_obj, failed_obj]
            ):
                with patch.object(backend._core, "_rawdev", return_value=raw_dev):
                    with pytest.raises(OSError, match="read failed"):
                        backend.batched_get_blocking([key1, key2])

            assert loaded_obj.get_ref_count() == 0
            assert failed_obj.get_ref_count() == 0
            assert backend.inflight_io_count() == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_rejects_when_full(memory_allocator, loop_in_thread):
    """Test that raw-block writes fail when no slot has been freed."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_evict",
        )
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.capacity_bytes": 3 * 4 * 1024 * 1024,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.slot_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")

            k1 = CacheEngineKey("test_model", 1, 0, 1, torch.bfloat16)
            k2 = CacheEngineKey("test_model", 1, 0, 2, torch.bfloat16)
            k3 = CacheEngineKey("test_model", 1, 0, 3, torch.bfloat16)

            o1 = alloc.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            o2 = alloc.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            o3 = alloc.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert o1 and o2 and o3
            assert (
                o1.tensor is not None
                and o2.tensor is not None
                and o3.tensor is not None
            )
            o1.tensor.fill_(1)
            o2.tensor.fill_(2)
            o3.tensor.fill_(3)

            f1 = backend.batched_submit_put_task([k1], [o1])[0]
            f2 = backend.batched_submit_put_task([k2], [o2])[0]
            f1.result(timeout=10)
            f2.result(timeout=10)

            assert backend.get_blocking(k1) is not None

            f3 = backend.batched_submit_put_task([k3], [o3])[0]
            with pytest.raises(RuntimeError, match="Failed to persist raw-block key"):
                f3.result(timeout=10)

            assert backend.get_blocking(k1) is not None
            assert backend.get_blocking(k2) is not None
            assert backend.contains(k3) is False
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_device_checkpoint_roundtrip(
    memory_allocator, loop_in_thread
):
    """Test on-device metadata checkpoint persistence across backend restarts."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        base_cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_manifest",
        )
        base_cfg.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=base_cfg,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend1 = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        alloc = AdHocMemoryAllocator(device="cpu")
        k1 = CacheEngineKey("test_model", 1, 0, 111, torch.bfloat16)
        o1 = alloc.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert o1 and o1.tensor is not None
        o1.tensor.fill_(9)
        expected = bytes(o1.byte_array)
        try:
            fut = backend1.batched_submit_put_task([k1], [o1])[0]
            fut.result(timeout=10)
        finally:
            backend1.close()

        # New backend instance should restore index and retrieve
        backend2 = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        try:
            assert backend2.contains(k1)
            out = backend2.get_blocking(k1)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend2.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_skips_checkpoint_on_init(
    memory_allocator, loop_in_thread
):
    """Skip on-device metadata checkpoint loading when configured."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_skip_checkpoint",
        )
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend1 = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        allocator = AdHocMemoryAllocator(device="cpu")
        key = CacheEngineKey("test_model", 1, 0, 222, torch.bfloat16)
        obj = allocator.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert obj is not None
        try:
            fut = backend1.batched_submit_put_task([key], [obj])[0]
            fut.result(timeout=10)
        finally:
            backend1.close()

        config.extra_config["rust_raw_block.load_checkpoint_on_init"] = False
        with patch.object(
            RawBlockCore,
            "_select_latest_checkpoint",
            side_effect=AssertionError("checkpoint retrieval should be skipped"),
        ) as mock_select_latest_checkpoint:
            backend2 = RustRawBlockBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu,
                loop=loop_in_thread,
                dst_device="cpu",
            )
        try:
            assert mock_select_latest_checkpoint.call_count == 0
            assert backend2.contains(key) is False
        finally:
            backend2.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_data_offsets_start_after_metadata(
    memory_allocator, loop_in_thread
):
    """Slot allocations must begin after reserved metadata region."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_offsets",
        )
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 8 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 777, torch.bfloat16)
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None
            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)

            entry_offset = backend.entry_offset(key)
            assert entry_offset is not None
            assert entry_offset >= 8 * 1024 * 1024
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_ignores_torn_newer_checkpoint(
    memory_allocator, loop_in_thread
):
    """
    If a newer checkpoint copy is torn, loader falls back to the older valid copy.
    """
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        base_cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_torn_checkpoint",
        )
        meta_total = 4 * 1024 * 1024
        align = 4096
        base_cfg.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": align,
            "rust_raw_block.header_bytes": align,
            "rust_raw_block.meta_total_bytes": meta_total,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=base_cfg,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )

        backend1 = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        alloc = AdHocMemoryAllocator(device="cpu")
        key = CacheEngineKey("test_model", 1, 0, 888, torch.bfloat16)
        obj = alloc.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert obj is not None and obj.tensor is not None
        obj.tensor.fill_(11)
        expected = bytes(obj.byte_array)
        try:
            fut = backend1.batched_submit_put_task([key], [obj])[0]
            fut.result(timeout=10)
        finally:
            torn_offset = backend1.metadata_container_offsets()[1]
            backend1.close()

        # Corrupt the newer checkpoint copy with invalid CRC.
        # Header format: <8sIQQI (magic, version, seq, payload_len, crc).
        header = struct.pack(
            "<8sIQQI", _DEFAULT_META_MAGIC, _DEFAULT_META_VERSION, 9999, 2, 0
        )
        padded_header = header + bytes(align - len(header))
        with open(dev_path, "r+b") as f:
            f.seek(torn_offset + align)
            f.write(b"{}")
            f.seek(torn_offset)
            f.write(padded_header)

        backend2 = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        try:
            assert backend2.contains(key)
            out = backend2.get_blocking(key)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend2.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_skips_invalid_checkpoint_entries(
    memory_allocator, loop_in_thread
):
    """Checkpoint restore should reject invalid offset/size metadata entries."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        base_cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_invalid_checkpoint",
        )
        base_cfg.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
            "rust_raw_block.meta_verify_on_load": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=base_cfg,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        try:
            entries = {}
            for chunk_hash, (offset, size) in {
                1: (backend.data_base_offset - backend.slot_bytes, 1024),
                2: (backend.data_base_offset + 1, 1024),
                3: (
                    backend.data_base_offset,
                    backend.slot_bytes - backend.header_bytes + 1,
                ),
            }.items():
                key = CacheEngineKey("test_model", 1, 0, chunk_hash, torch.bfloat16)
                entries[key.to_string()] = {
                    "offset": offset,
                    "size": size,
                    "shape": [2, 16, 8, 128],
                    "dtype": "bfloat16",
                    "fmt": MemoryFormat.KV_T2D.name,
                    "cached_positions": None,
                }

            applied = backend.apply_loaded_state(
                {
                    "version": 1,
                    "device_path": dev_path,
                    "capacity_bytes": backend.capacity_bytes,
                    "block_align": backend.block_align,
                    "header_bytes": backend.header_bytes,
                    "slot_bytes": backend.slot_bytes,
                    "meta_total_bytes": backend.meta_total_bytes,
                    "meta_magic": backend.meta_magic_text,
                    "meta_version": backend.meta_version,
                    "data_base_offset": backend.data_base_offset,
                    "next_slot": 0,
                    "free_slots": [],
                    "entries": entries,
                }
            )
            assert applied is True
            assert backend.indexed_key_count() == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_recovers_legacy_key_dtype(
    memory_allocator, loop_in_thread
):
    """Checkpoint recovery should fall back to dtype encoded in legacy keys."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        base_cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_legacy_dtype",
        )
        base_cfg.storage_plugins = []
        base_cfg.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.slot_bytes": 1 * 1024 * 1024,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
            "rust_raw_block.meta_verify_on_load": False,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=base_cfg,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        try:
            key = CacheEngineKey("test_model", 1, 0, 99, torch.bfloat16)
            applied = backend.apply_loaded_state(
                {
                    "version": 1,
                    "device_path": dev_path,
                    "capacity_bytes": backend.capacity_bytes,
                    "block_align": backend.block_align,
                    "header_bytes": backend.header_bytes,
                    "slot_bytes": backend.slot_bytes,
                    "meta_total_bytes": backend.meta_total_bytes,
                    "meta_magic": backend.meta_magic_text,
                    "meta_version": backend.meta_version,
                    "data_base_offset": backend.data_base_offset,
                    "next_slot": 1,
                    "free_slots": [],
                    # Older checkpoints may include this key; recovery ignores it.
                    "lru_keys": [key.to_string()],
                    "entries": {
                        key.to_string(): {
                            "offset": backend.data_base_offset,
                            "size": 1024,
                            "shape": [512],
                            "dtype": "torch.bfloat16",
                            "fmt": MemoryFormat.KV_2LTD.name,
                            "cached_positions": None,
                        }
                    },
                }
            )

            assert applied is True
            loaded = backend.batched_get_blocking([key])
            assert loaded[0] is not None
            assert loaded[0].metadata.dtype is torch.bfloat16
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_tp4_initialization(memory_allocator, loop_in_thread):
    """Test TP=4 initialization with per-TP device paths."""
    TP = 4
    with tempfile.TemporaryDirectory() as td:
        device_paths = [os.path.join(td, f"device{i}.bin") for i in range(TP)]
        for p in device_paths:
            with open(p, "wb") as f:
                f.truncate(256 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_tp4_init",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.per_tp_device_paths": {
                str(i): device_paths[i] for i in range(TP)
            },
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }

        backends = []
        try:
            for i in range(TP):
                metadata = LMCacheMetadata(
                    model_name="test-model",
                    world_size=TP,
                    local_world_size=TP,
                    worker_id=i,
                    local_worker_id=i,
                    kv_dtype=torch.bfloat16,
                    kv_shape=(32, 2, 256, 32, 128),
                )
                local_cpu = LocalCPUBackend(
                    config=config,
                    metadata=metadata,
                    dst_device="cpu",
                    memory_allocator=memory_allocator,
                )
                be = RustRawBlockBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=local_cpu,
                    loop=loop_in_thread,
                    dst_device="cpu",
                )
                backends.append(be)
                assert be.device_path == device_paths[i]
            assert len({b.device_path for b in backends}) == TP
        finally:
            for backend in backends:
                backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_tp4_initialization_accepts_integer_yaml_keys(
    memory_allocator, loop_in_thread
):
    """Accept integer per-TP YAML keys in addition to quoted string keys."""
    TP = 4
    with tempfile.TemporaryDirectory() as td:
        device_paths = [os.path.join(td, f"device{i}.bin") for i in range(TP)]
        for p in device_paths:
            with open(p, "wb") as f:
                f.truncate(256 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_tp4_init_int_keys",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.per_tp_device_paths": {
                i: device_paths[i] for i in range(TP)
            },
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }

        backends = []
        try:
            for i in range(TP):
                metadata = LMCacheMetadata(
                    model_name="test-model",
                    world_size=TP,
                    local_world_size=TP,
                    worker_id=i,
                    local_worker_id=i,
                    kv_dtype=torch.bfloat16,
                    kv_shape=(32, 2, 256, 32, 128),
                )
                local_cpu = LocalCPUBackend(
                    config=config,
                    metadata=metadata,
                    dst_device="cpu",
                    memory_allocator=memory_allocator,
                )
                be = RustRawBlockBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=local_cpu,
                    loop=loop_in_thread,
                    dst_device="cpu",
                )
                backends.append(be)
                assert be.device_path == device_paths[i]
        finally:
            for backend in backends:
                backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_tp4_comprehensive_io(memory_allocator, loop_in_thread):
    """Comprehensive TP=4 I/O test covering roundtrip, multiple ops, and isolation."""
    TP = 4
    with tempfile.TemporaryDirectory() as td:
        device_paths = [os.path.join(td, f"device{i}.bin") for i in range(TP)]
        for p in device_paths:
            with open(p, "wb") as f:
                f.truncate(1024 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_tp4_io",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.per_tp_device_paths": {
                str(i): device_paths[i] for i in range(TP)
            },
            "rust_raw_block.capacity_bytes": 0,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }

        backends = []
        try:
            for i in range(TP):
                metadata = LMCacheMetadata(
                    model_name="test-model",
                    world_size=TP,
                    local_world_size=TP,
                    worker_id=i,
                    local_worker_id=i,
                    kv_dtype=torch.bfloat16,
                    kv_shape=(32, 2, 256, 32, 128),
                )
                local_cpu = LocalCPUBackend(
                    config=config,
                    metadata=metadata,
                    dst_device="cpu",
                    memory_allocator=memory_allocator,
                )
                be = RustRawBlockBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=local_cpu,
                    loop=loop_in_thread,
                    dst_device="cpu",
                )
                backends.append(be)

            allocator = AdHocMemoryAllocator(device="cpu")
            for tp in range(TP):
                backend = backends[tp]
                key = CacheEngineKey("test-model", TP, tp, 1000 + tp, torch.bfloat16)
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None
                obj.tensor.fill_(tp * 10)
                expected = bytes(obj.byte_array)

                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

                out = backend.get_blocking(key)
                assert out is not None
                assert bytes(out.byte_array) == expected
                out.ref_count_down()

                for other in range(TP):
                    if other == tp:
                        continue
                    other_key = CacheEngineKey(
                        "test-model", TP, other, 1000 + other, torch.bfloat16
                    )
                    assert backend.get_blocking(other_key) is None

            all_keys = []
            for tp in range(TP):
                keys = []
                for i in range(TP - 1):
                    allocator = AdHocMemoryAllocator(device="cpu")
                    obj = allocator.allocate(
                        [torch.Size([2, 16, 8, 128])],
                        [torch.bfloat16],
                        fmt=MemoryFormat.KV_T2D,
                    )
                    assert obj is not None
                    assert obj.tensor is not None
                    obj.tensor.fill_(tp * 100 + i)
                    key = CacheEngineKey(
                        "test-model", TP, tp, tp * 100 + i, torch.bfloat16
                    )
                    keys.append(key)
                    futs = backends[tp].batched_submit_put_task([key], [obj])
                    assert futs is not None
                    futs[0].result(timeout=10)
                    obj.ref_count_down()
                all_keys.append(keys)

            for tp in range(TP):
                for key in all_keys[tp]:
                    out = backends[tp].get_blocking(key)
                    assert out is not None
                    assert out.tensor is not None
                    out.ref_count_down()

            for tp in range(TP):
                for other in range(TP):
                    if other == tp:
                        continue
                    for key in all_keys[other]:
                        assert backends[tp].get_blocking(key) is None
        finally:
            for backend in backends:
                backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_tp_paths_must_be_unique(
    memory_allocator, loop_in_thread
):
    """Reject TP config when multiple ranks point to the same partition."""
    with tempfile.TemporaryDirectory() as td:
        shared_dev = os.path.join(td, "shared.bin")
        with open(shared_dev, "wb") as f:
            f.truncate(512 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_tp_dupe_paths",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.per_tp_device_paths": {"0": shared_dev, "1": shared_dev},
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test-model",
            world_size=2,
            local_world_size=2,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(32, 2, 256, 32, 128),
        )
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        with pytest.raises(ValueError, match="Duplicate device path configured"):
            RustRawBlockBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu,
                loop=loop_in_thread,
                dst_device="cpu",
            )


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_warns_on_cross_rank_metadata_load(
    memory_allocator, loop_in_thread
):
    """Warn when loading metadata whose first entry belongs to another worker."""
    with tempfile.TemporaryDirectory() as td:
        device0 = os.path.join(td, "device0.bin")
        device1 = os.path.join(td, "device1.bin")
        for p in [device0, device1]:
            with open(p, "wb") as f:
                f.truncate(512 * 1024 * 1024)

        base_extra = {
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }

        config_tp0 = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_cross_rank_warn_tp0",
        )
        config_tp0.storage_plugins = []
        config_tp0.extra_config = {
            **base_extra,
            "rust_raw_block.per_tp_device_paths": {"0": device0, "1": device1},
        }
        metadata_tp0 = LMCacheMetadata(
            model_name="test-model",
            world_size=2,
            local_world_size=2,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(32, 2, 256, 32, 128),
        )
        local_cpu_tp0 = LocalCPUBackend(
            config=config_tp0,
            metadata=metadata_tp0,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend_tp0 = RustRawBlockBackend(
            config=config_tp0,
            metadata=metadata_tp0,
            local_cpu_backend=local_cpu_tp0,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        key_tp0 = CacheEngineKey("test-model", 2, 0, 31337, torch.bfloat16)
        allocator = AdHocMemoryAllocator(device="cpu")
        obj = allocator.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert obj is not None
        assert obj.tensor is not None
        obj.tensor.fill_(9)
        try:
            futs = backend_tp0.batched_submit_put_task([key_tp0], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()
        finally:
            backend_tp0.close()

        config_tp1_mis = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_cross_rank_warn_tp1",
        )
        config_tp1_mis.storage_plugins = []
        config_tp1_mis.extra_config = {
            **base_extra,
            "rust_raw_block.per_tp_device_paths": {"1": device0},
        }
        metadata_tp1 = LMCacheMetadata(
            model_name="test-model",
            world_size=2,
            local_world_size=2,
            worker_id=1,
            local_worker_id=1,
            kv_dtype=torch.bfloat16,
            kv_shape=(32, 2, 256, 32, 128),
        )
        local_cpu_tp1 = LocalCPUBackend(
            config=config_tp1_mis,
            metadata=metadata_tp1,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )

        with patch(
            "lmcache.v1.storage_backend.plugins.rust_raw_block_backend.logger.warning"
        ) as mock_warning:
            backend_tp1 = RustRawBlockBackend(
                config=config_tp1_mis,
                metadata=metadata_tp1,
                local_cpu_backend=local_cpu_tp1,
                loop=loop_in_thread,
                dst_device="cpu",
            )
        try:
            matched = False
            for call in mock_warning.call_args_list:
                call_args = call.args
                if not call_args:
                    continue
                fmt = call_args[0]
                if "loaded metadata may belong to another worker" not in str(fmt):
                    continue
                assert int(call_args[2]) == 1
                assert int(call_args[3]) == 0
                matched = True
                break
            assert matched, "Expected cross-rank metadata warning was not emitted"
        finally:
            backend_tp1.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_uring_put_get_roundtrip(
    memory_allocator, loop_in_thread
):
    """Test batched write with io_uring and verify data integrity on read.

    Writes 128 items asynchronously, then reads them back and verifies
    the data matches what was written.
    """
    NUM_OPS = 128

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(1024 * 1024 * 1024)  # 1G

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=1,
            lmcache_instance_id="test_rust_raw_block_uring",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.use_uring": True,  # Enable io_uring
            "rust_raw_block.use_odirect": True,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
            chunk_size=256,
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = local_cpu.memory_allocator

            # Create keys and memory objects with unique data patterns
            keys = []
            objs = []
            expected_data = []

            for i in range(NUM_OPS):
                key = CacheEngineKey("test_model", 1, 0, i, torch.bfloat16)
                keys.append(key)

                # Each object gets a unique fill value
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None

                # Use unique pattern: (i + 1) as fill value
                obj.tensor.fill_(float(i + 1))
                expected_data.append(bytes(obj.byte_array))
                objs.append(obj)

            # Submit all writes using io_uring batched write
            futs = backend.batched_submit_put_task(keys, objs)
            assert futs is not None
            futs[0].result(timeout=10)

            # Read back and verify data integrity
            for i, key in enumerate(keys):
                out = backend.get_blocking(key)
                assert out is not None, f"Failed to read key {i}"
                actual_data = bytes(out.byte_array)
                # Use `raise AssertionError` instead of `assert ==` so pytest
                # doesn't try to render a difflib diff between two large
                # byte arrays on failure (effectively a hang on _fancy_replace).
                if actual_data != expected_data[i]:
                    raise AssertionError(
                        f"Data mismatch for key {i}: "
                        f"expected first bytes {expected_data[i][:16]!r}, "
                        f"got {actual_data[:16]!r}"
                    )

        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_close_with_inflight(memory_allocator, loop_in_thread):
    """Test that RawBlockDevice close handles queued / inflight io_uring writes."""
    del memory_allocator, loop_in_thread
    # Third Party
    from lmcache_rust_raw_block_io import RawBlockDevice

    NUM_OPS = 128

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(16 * 1024 * 1024)

        writer = RawBlockDevice(
            dev_path,
            writable=True,
            use_odirect=False,
            alignment=4096,
            io_engine="io_uring",
        )
        payload_len = 4096
        offsets: list[int] = []
        buffers: list[bytearray] = []
        total_lens: list[int] = []

        for i in range(NUM_OPS):
            payload = bytearray([i % 251]) * payload_len
            offsets.append(i * payload_len)
            buffers.append(payload)
            total_lens.append(payload_len)

        writer.batched_write(offsets, buffers, total_lens)
        time.sleep(0.001)
        writer.close()

        reader = RawBlockDevice(
            dev_path,
            writable=False,
            use_odirect=False,
            alignment=4096,
            io_engine="posix",
        )
        try:
            for i, expected in enumerate(buffers):
                actual = bytearray(payload_len)
                reader.pread_into(i * payload_len, actual, payload_len, payload_len)
                # Use `raise AssertionError` instead of `assert ==` so pytest
                # doesn't try to render a difflib-based diff between two
                # large byte arrays on failure, which is O(n^2) on
                # `_fancy_replace` and effectively never returns.
                if actual != expected:
                    raise AssertionError(
                        f"Data mismatch at offset {i * payload_len}: "
                        f"expected first bytes {bytes(expected[:16])!r}, "
                        f"got {bytes(actual[:16])!r}"
                    )
        finally:
            reader.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_with_uring(
    memory_allocator, loop_in_thread
):
    """Test batched_get_blocking and batched_get_non_blocking with io_uring enabled."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_batched_get_uring",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
            "rust_raw_block.use_uring": True,  # Enable io_uring
            "rust_raw_block.use_odirect": True,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = local_cpu.memory_allocator
            key1 = CacheEngineKey("test_model", 1, 0, 4001, torch.bfloat16)
            key2 = CacheEngineKey("test_model", 1, 0, 4002, torch.bfloat16)
            key_miss = CacheEngineKey("test_model", 1, 0, 4003, torch.bfloat16)
            key3 = CacheEngineKey("test_model", 1, 0, 4004, torch.bfloat16)

            obj1 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            obj2 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            obj3 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj1 is not None and obj1.tensor is not None
            assert obj2 is not None and obj2.tensor is not None
            assert obj3 is not None and obj3.tensor is not None
            obj1.tensor.fill_(41)
            obj2.tensor.fill_(42)
            obj3.tensor.fill_(43)
            expected1 = bytes(obj1.byte_array)
            expected2 = bytes(obj2.byte_array)

            # Put keys 1, 2, and 3
            for key, obj in ((key1, obj1), (key2, obj2), (key3, obj3)):
                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

            # Test batched_get_blocking with uring. It should stop at first miss
            blocking_results = backend.batched_get_blocking(
                [key1, key2, key_miss, key3]
            )
            assert len(blocking_results) == 4
            assert blocking_results[0] is not None
            assert bytes(blocking_results[0].byte_array) == expected1
            assert blocking_results[1] is not None
            assert bytes(blocking_results[1].byte_array) == expected2
            assert blocking_results[2] is None  # Miss
            assert blocking_results[3] is None  # After miss
            blocking_results[0].ref_count_down()
            blocking_results[1].ref_count_down()

            # Test batched_get_non_blocking with uring.
            # It should return only successful prefix
            future = asyncio.run_coroutine_threadsafe(
                backend.batched_get_non_blocking(
                    "lookup-uring", [key1, key2, key_miss, key3]
                ),
                loop_in_thread,
            )
            async_results = future.result(timeout=10)
            assert len(async_results) == 2  # Only key1 and key2
            assert bytes(async_results[0].byte_array) == expected1
            assert bytes(async_results[1].byte_array) == expected2
            async_results[0].ref_count_down()
            async_results[1].ref_count_down()

        finally:
            backend.close()


def _make_raw_block_backend(
    dev_path: str,
    memory_allocator: Any,
    loop: asyncio.AbstractEventLoop,
    *,
    io_engine: str = "io_uring",
) -> RustRawBlockBackend:
    """Build a RustRawBlockBackend over a fake raw-block device.

    Args:
        dev_path: Backing device path passed to the (faked) raw block device.
        memory_allocator: Allocator shared with the local CPU backend.
        loop: Running asyncio event loop for dispatched put coroutines.
        io_engine: Raw-block I/O engine to configure.

    Returns:
        A backend whose core reports the requested I/O engine.
    """
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="test_rust_raw_block_backend_plugin_dedup",
    )
    config.storage_plugins = []
    config.extra_config = {
        "rust_raw_block.device_path": dev_path,
        "rust_raw_block.block_align": 4096,
        "rust_raw_block.header_bytes": 4096,
        "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
        "rust_raw_block.meta_enable_periodic": False,
        "rust_raw_block.io_engine": io_engine,
    }
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )
    local_cpu = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
        memory_allocator=memory_allocator,
    )
    return RustRawBlockBackend(
        config=config,
        metadata=metadata,
        local_cpu_backend=local_cpu,
        loop=loop,
        dst_device="cpu",
    )


def test_rust_raw_block_backend_batched_submit_rolls_back_refs_on_dispatch_failure(
    monkeypatch, memory_allocator, loop_in_thread
):
    """A dispatch failure rolls back every ref_count_up and clears put_tasks.

    The plugin raises the ref count for each pending object before handing the
    batch to the event loop. If that hand-off raises (e.g. the loop is shutting
    down), every raised ref must be returned and every key removed from the
    in-flight put-task set so nothing leaks.
    """
    _install_fake_raw_block_device(monkeypatch, size_bytes=64 * 1024 * 1024)
    backend = _make_raw_block_backend(
        "/tmp/plugin-rollback", memory_allocator, loop_in_thread
    )
    try:
        allocator = AdHocMemoryAllocator(device="cpu")
        key1 = CacheEngineKey("test_model", 1, 0, 2001, torch.bfloat16)
        key2 = CacheEngineKey("test_model", 1, 0, 2002, torch.bfloat16)
        obj1 = allocator.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        obj2 = allocator.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert obj1 is not None and obj2 is not None
        before1 = obj1.get_ref_count()
        before2 = obj2.get_ref_count()

        with patch(
            "lmcache.v1.storage_backend.plugins.rust_raw_block_backend."
            "asyncio.run_coroutine_threadsafe",
            side_effect=RuntimeError("event loop is shutting down"),
        ):
            with pytest.raises(RuntimeError):
                backend.batched_submit_put_task([key1, key2], [obj1, obj2])

        assert obj1.get_ref_count() == before1
        assert obj2.get_ref_count() == before2
        assert not backend.exists_in_put_tasks(key1)
        assert not backend.exists_in_put_tasks(key2)
        obj1.ref_count_down()
        obj2.ref_count_down()
    finally:
        backend.close()


def test_rust_raw_block_backend_batched_submit_rolls_back_only_unscheduled_refs(
    monkeypatch, memory_allocator, loop_in_thread
):
    """A partial per-key dispatch failure leaves scheduled task cleanup to task."""
    _install_fake_raw_block_device(monkeypatch, size_bytes=64 * 1024 * 1024)
    backend = _make_raw_block_backend(
        "/tmp/plugin-partial-rollback",
        memory_allocator,
        loop_in_thread,
        io_engine="posix",
    )
    try:
        allocator = AdHocMemoryAllocator(device="cpu")
        key1 = CacheEngineKey("test_model", 1, 0, 3001, torch.bfloat16)
        key2 = CacheEngineKey("test_model", 1, 0, 3002, torch.bfloat16)
        obj1 = allocator.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        obj2 = allocator.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert obj1 is not None and obj2 is not None
        before1 = obj1.get_ref_count()
        before2 = obj2.get_ref_count()
        real_run_coroutine_threadsafe = asyncio.run_coroutine_threadsafe
        first_future: Future | None = None

        def fake_put_many(
            keys: list[RawBlockKeySpec],
            objs: list[Any],
        ) -> RawBlockPutManyResult:
            del objs
            return RawBlockPutManyResult(
                results=[True] * len(keys),
                stored_keys=[key.encoded for key in keys],
            )

        def run_first_then_fail(
            coro: Any,
            loop: asyncio.AbstractEventLoop,
        ) -> Future:
            nonlocal first_future
            if first_future is None:
                first_future = real_run_coroutine_threadsafe(coro, loop)
                return first_future
            raise RuntimeError("event loop is shutting down")

        with (
            patch.object(backend._core, "put_many", side_effect=fake_put_many),
            patch(
                "lmcache.v1.storage_backend.plugins.rust_raw_block_backend."
                "asyncio.run_coroutine_threadsafe",
                side_effect=run_first_then_fail,
            ),
        ):
            with pytest.raises(RuntimeError):
                backend.batched_submit_put_task([key1, key2], [obj1, obj2])

            assert first_future is not None
            first_future.result(timeout=10)

        assert obj1.get_ref_count() == before1
        assert obj2.get_ref_count() == before2
        assert not backend.exists_in_put_tasks(key1)
        assert not backend.exists_in_put_tasks(key2)
        obj1.ref_count_down()
        obj2.ref_count_down()
    finally:
        backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_batched_write_rejects_misaligned_offset():
    """batched_write raises ValueError for a misaligned offset when use_odirect=True."""
    # Third Party
    from lmcache_rust_raw_block_io import RawBlockDevice

    align = 4096
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        dev = RawBlockDevice(
            dev_path,
            writable=True,
            use_odirect=True,
            alignment=align,
            io_engine="io_uring",
        )
        try:
            buf = bytearray(align)
            with pytest.raises(ValueError, match="aligned offset"):
                dev.batched_write([1], [buf], [align])  # offset 1 is not aligned
        finally:
            dev.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_batched_write_rejects_misaligned_total_len():
    """batched_write rejects misaligned total_len with O_DIRECT enabled."""
    # Third Party
    from lmcache_rust_raw_block_io import RawBlockDevice

    align = 4096
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        dev = RawBlockDevice(
            dev_path,
            writable=True,
            use_odirect=True,
            alignment=align,
            io_engine="io_uring",
        )
        try:
            buf = bytearray(align)
            with pytest.raises(ValueError, match="aligned total_len"):
                dev.batched_write([0], [buf], [1])  # total_len 1 is not aligned
        finally:
            dev.close()
