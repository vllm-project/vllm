# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import tempfile
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.plugins.dax_backend import DaxBackend
import lmcache.v1.storage_backend.plugins.dax_backend as dax_backend_module


@pytest.fixture
def loop_in_thread() -> Generator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def _create_metadata(
    chunk_size: int = 16,
    world_size: int = 1,
    role: str = "worker",
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, chunk_size, 2, 8),
        role=role,
    )


def _create_multi_group_metadata(chunk_size: int = 16) -> LMCacheMetadata:
    # First Party
    import lmcache.c_ops as lmc_ops

    metadata = _create_metadata(chunk_size=chunk_size)
    # Two single-layer groups whose only differing signature field is
    # head_size (8 vs 16), exercising the multi-group code path.
    kv_caches = [
        torch.empty(2, 1, chunk_size, 1, 8, dtype=torch.bfloat16),
        torch.empty(2, 1, chunk_size, 1, 16, dtype=torch.bfloat16),
    ]
    metadata.kv_layer_groups_manager = KVLayerGroupsManager(
        kv_caches,
        [lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS] * len(kv_caches),
    )
    return metadata


def _create_config(
    *,
    chunk_size: int = 16,
    local_cpu: bool = True,
    max_local_cpu_size: float = 0.1,
    extra_config: dict | None = None,
    storage_plugins: list[str] | None = None,
) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=local_cpu,
        max_local_cpu_size=max_local_cpu_size,
        lmcache_instance_id="test_dax_backend",
    )
    merged_extra_config = {
        "dax.restore_workers": 1,
        "dax.restore_max_regions": 1,
        "dax.retrieve_staging_slab_bytes": 8 * 1024 * 1024,
    }
    if extra_config is not None:
        merged_extra_config.update(extra_config)
    config.extra_config = merged_extra_config
    if storage_plugins is not None:
        config.storage_plugins = storage_plugins
    return config


def _allocate_kv_obj(
    *,
    num_tokens: int,
    hidden_dim: int = 8,
    fill_value: int = 0,
) -> MemoryObj:
    alloc = AdHocMemoryAllocator(device="cpu")
    obj = alloc.allocate(
        [torch.Size([2, num_tokens, hidden_dim])],
        [torch.bfloat16],
        fmt=MemoryFormat.KV_T2D,
    )
    assert obj is not None
    assert obj.tensor is not None
    obj.tensor.fill_(fill_value)
    return obj


def _store_tensor(
    backend: DaxBackend,
    key: CacheEngineKey,
    *,
    num_tokens: int,
    hidden_dim: int = 8,
    fill_value: int = 0,
) -> None:
    obj = _allocate_kv_obj(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        fill_value=fill_value,
    )
    try:
        futures = backend.batched_submit_put_task([key], [obj])
        if futures:
            for future in futures:
                future.result(timeout=5)
    finally:
        obj.ref_count_down()


def test_dax_backend_roundtrip(memory_allocator, loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 1, torch.bfloat16)
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(5)

            futs = backend.batched_submit_put_task([key], [obj])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)

            out = backend.get_blocking(key)
            assert out is not None
            assert out.tensor is not None
            assert torch.equal(out.tensor, obj.tensor)
            out.ref_count_down()
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_batched_get_blocking_keeps_positional_holes(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key1 = CacheEngineKey("test_model", 1, 0, 11, torch.bfloat16)
            key2 = CacheEngineKey("test_model", 1, 0, 12, torch.bfloat16)
            key3 = CacheEngineKey("test_model", 1, 0, 13, torch.bfloat16)
            _store_tensor(backend, key1, num_tokens=16, fill_value=3)
            _store_tensor(backend, key3, num_tokens=16, fill_value=7)

            results = backend.batched_get_blocking([key1, key2, key3])
            assert len(results) == 3
            assert results[1] is None
            assert results[0] is not None
            assert results[2] is not None
            assert torch.all(results[0].tensor == 3)
            assert torch.all(results[2].tensor == 7)
            results[0].ref_count_down()
            results[2].ref_count_down()
        finally:
            backend.close()


def test_dax_backend_batched_get_blocking_passes_cached_fmt_to_allocator(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        seen_formats: list[MemoryFormat] = []
        original_batched_allocate = local_cpu.batched_allocate

        def _tracking_batched_allocate(
            shapes: torch.Size | list[torch.Size],
            dtypes: torch.dtype | list[torch.dtype],
            batch_size: int,
            fmt: MemoryFormat | None = None,
            eviction: bool = True,
            busy_loop: bool = True,
        ) -> list[MemoryObj] | None:
            assert fmt is not None
            seen_formats.append(fmt)
            return original_batched_allocate(
                shapes,
                dtypes,
                batch_size,
                fmt,
                eviction=eviction,
                busy_loop=busy_loop,
            )

        monkeypatch.setattr(local_cpu, "batched_allocate", _tracking_batched_allocate)

        try:
            key1 = CacheEngineKey("test_model", 1, 0, 14, torch.bfloat16)
            key2 = CacheEngineKey("test_model", 1, 0, 15, torch.bfloat16)
            _store_tensor(backend, key1, num_tokens=16, fill_value=4)
            _store_tensor(backend, key2, num_tokens=16, fill_value=8)

            results = backend.batched_get_blocking([key1, key2])
            assert seen_formats == [MemoryFormat.KV_T2D]
            for result in results:
                assert result is not None
                result.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_batched_get_blocking_handles_heterogeneous_chunk_shapes(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            full_key = CacheEngineKey("test_model", 1, 0, 21, torch.bfloat16)
            tail_key = CacheEngineKey("test_model", 1, 0, 22, torch.bfloat16)
            _store_tensor(backend, full_key, num_tokens=16, fill_value=5)
            _store_tensor(backend, tail_key, num_tokens=8, fill_value=9)

            results = backend.batched_get_blocking([full_key, tail_key])
            assert [result.get_shape() for result in results if result is not None] == [
                torch.Size([2, 16, 8]),
                torch.Size([2, 8, 8]),
            ]
            assert results[0] is not None
            assert results[1] is not None
            assert torch.all(results[0].tensor == 5)
            assert torch.all(results[1].tensor == 9)
            for result in results:
                assert result is not None
                result.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_batched_get_non_blocking_stops_at_first_miss(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key1 = CacheEngineKey("test_model", 1, 0, 31, torch.bfloat16)
            key2 = CacheEngineKey("test_model", 1, 0, 32, torch.bfloat16)
            key3 = CacheEngineKey("test_model", 1, 0, 33, torch.bfloat16)
            _store_tensor(backend, key1, num_tokens=16, fill_value=1)
            _store_tensor(backend, key3, num_tokens=16, fill_value=2)

            results = asyncio.run(
                backend.batched_get_non_blocking("lookup", [key1, key2, key3])
            )
            assert len(results) == 1
            assert torch.all(results[0].tensor == 1)
            results[0].ref_count_down()
        finally:
            backend.close()


def test_dax_backend_blocking_and_async_share_restore_dispatch(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        key1 = CacheEngineKey("test_model", 1, 0, 43, torch.bfloat16)
        key2 = CacheEngineKey("test_model", 1, 0, 44, torch.bfloat16)
        first_restore_started = threading.Event()
        allow_first_restore = threading.Event()
        second_restore_started = threading.Event()
        original_restore_batch = backend._restore_batch
        call_count = 0
        call_count_lock = threading.Lock()

        def _gated_restore_batch(keys, prefix_only):
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                first_restore_started.set()
                assert allow_first_restore.wait(timeout=1)
            elif current_call == 2:
                second_restore_started.set()
            return original_restore_batch(keys, prefix_only)

        def _read_async() -> list[MemoryObj]:
            return asyncio.run(backend.batched_get_non_blocking("lookup", [key2]))

        monkeypatch.setattr(backend, "_restore_batch", _gated_restore_batch)

        try:
            _store_tensor(backend, key1, num_tokens=16, fill_value=12)
            _store_tensor(backend, key2, num_tokens=16, fill_value=13)

            with ThreadPoolExecutor(max_workers=2) as executor:
                blocking_future = executor.submit(backend.batched_get_blocking, [key1])
                assert first_restore_started.wait(timeout=1)

                async_future = executor.submit(_read_async)
                assert not second_restore_started.wait(timeout=0.2)

                allow_first_restore.set()

                blocking_results = blocking_future.result(timeout=2)
                assert second_restore_started.wait(timeout=1)
                async_results = async_future.result(timeout=2)

            assert len(blocking_results) == 1
            assert len(async_results) == 1
            assert blocking_results[0] is not None
            assert torch.all(blocking_results[0].tensor == 12)
            assert torch.all(async_results[0].tensor == 13)
            blocking_results[0].ref_count_down()
            async_results[0].ref_count_down()
        finally:
            backend.close()


def test_dax_backend_batched_memcpy_helper() -> None:
    src = torch.arange(16, dtype=torch.uint8)
    dst = torch.zeros(16, dtype=torch.uint8)
    dst2 = torch.zeros(4, dtype=torch.uint8)

    dax_backend_module.lmc_ops.batched_memcpy(
        [src.data_ptr(), src.data_ptr() + 4],
        [dst.data_ptr(), dst2.data_ptr()],
        [8, 4],
    )

    assert torch.equal(dst[:8], src[:8])
    assert torch.equal(dst2, src[4:8])


def test_dax_backend_rejects_tp_gt_1(loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 4 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16, world_size=2)

        with pytest.raises(ValueError, match="only supports TP=1"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=loop_in_thread,
                dst_device="cpu",
            )


def test_dax_backend_requires_local_cpu_backend() -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)

        with pytest.raises(ValueError, match="requires local_cpu_backend"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=None,
                dst_device="cpu",
            )


def test_dax_backend_rejects_multi_group_metadata_at_init() -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(8 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 8 / 1024,
            },
        )
        metadata = _create_multi_group_metadata(chunk_size=16)

        with pytest.raises(ValueError, match="single-group KV layout"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=None,
                dst_device="cpu",
            )


def test_dax_backend_failed_init_does_not_leak_fds() -> None:
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("/proc/self/fd is not available on this platform")

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)

        fd_before = len(os.listdir("/proc/self/fd"))
        for _ in range(3):
            local_cpu = LocalCPUBackend(
                config=config,
                metadata=metadata,
                dst_device="cpu",
                memory_allocator=AdHocMemoryAllocator(device="cpu"),
            )
            with pytest.raises(RuntimeError, match="exceeds device capacity"):
                DaxBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=local_cpu,
                    loop=None,
                    dst_device="cpu",
                )
        fd_after = len(os.listdir("/proc/self/fd"))
        assert fd_after == fd_before


def test_dax_backend_oversized_put_raises_error(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 32 / (1024 * 1024),  # one slot
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            oversized_key = CacheEngineKey("test_model", 1, 0, 704, torch.bfloat16)
            oversized = alloc.allocate(
                [torch.Size([2, 1025, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert oversized is not None
            assert oversized.get_size() > backend.slot_bytes

            with pytest.raises(ValueError, match="exceeds slot size"):
                backend.batched_submit_put_task([oversized_key], [oversized])
            assert not backend.contains(oversized_key)
            oversized.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_put_rejects_mismatched_key_and_obj_lengths(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 64 / (1024 * 1024),
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None

            with pytest.raises(ValueError, match="same length"):
                backend.batched_submit_put_task(
                    [
                        CacheEngineKey("test_model", 1, 0, 801, torch.bfloat16),
                        CacheEngineKey("test_model", 1, 0, 802, torch.bfloat16),
                    ],
                    [obj],
                )
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_multi_tensor_put_skips_without_indexing_or_leaking_slots(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 32 / (1024 * 1024),  # one slot
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            multi_key = CacheEngineKey("test_model", 1, 0, 706, torch.bfloat16)
            multi = alloc.allocate(
                [torch.Size([2, 128, 8]), torch.Size([2, 128, 8])],
                [torch.bfloat16, torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert multi is not None

            backend.batched_submit_put_task([multi_key], [multi])
            assert not backend.contains(multi_key)
            assert backend.get_blocking(multi_key) is None
            multi.ref_count_down()

            valid_key = CacheEngineKey("test_model", 1, 0, 708, torch.bfloat16)
            reclaimed = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert reclaimed is not None
            backend.batched_submit_put_task([valid_key], [reclaimed])
            out = backend.get_blocking(valid_key)
            assert out is not None
            out.ref_count_down()
            reclaimed.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_get_blocking_releases_lock_during_read(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        key1 = CacheEngineKey("test_model", 1, 0, 701, torch.bfloat16)
        key2 = CacheEngineKey("test_model", 1, 0, 702, torch.bfloat16)
        read_started = threading.Event()
        allow_read = threading.Event()
        remove_finished = threading.Event()
        reader_result: list[MemoryObj | None] = []
        remove_result: list[bool] = []
        original_do_read = DaxBackend._do_read

        def _blocking_do_read(self, offset, memory_obj, size) -> None:
            read_started.set()
            assert allow_read.wait(timeout=1)
            original_do_read(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_read", _blocking_do_read)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            for key, fill_value in ((key1, 3), (key2, 4)):
                obj = alloc.allocate(
                    [torch.Size([2, 16, 8])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None
                obj.tensor.fill_(fill_value)
                backend.batched_submit_put_task([key], [obj])
                obj.ref_count_down()

            def _reader() -> None:
                reader_result.append(backend.get_blocking(key1))

            def _remover() -> None:
                remove_result.append(backend.remove(key2))
                remove_finished.set()

            reader = threading.Thread(target=_reader)
            reader.start()
            assert read_started.wait(timeout=1)

            remover = threading.Thread(target=_remover)
            remover.start()
            assert remove_finished.wait(timeout=0.2)
            assert remove_result[0] is True

            allow_read.set()
            reader.join(timeout=1)
            remover.join(timeout=1)
            assert not reader.is_alive()
            assert not remover.is_alive()

            result = reader_result[0]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 3)
            result.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_batched_get_blocking_releases_lock_during_restore(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        key1 = CacheEngineKey("test_model", 1, 0, 711, torch.bfloat16)
        key2 = CacheEngineKey("test_model", 1, 0, 712, torch.bfloat16)
        restore_started = threading.Event()
        allow_restore = threading.Event()
        remove_finished = threading.Event()
        reader_result: list[list[MemoryObj | None]] = []
        remove_result: list[bool] = []
        original_batched_memcpy = backend._batched_memcpy
        call_count = 0

        def _blocking_batched_memcpy(src_ptrs, dst_ptrs, sizes) -> None:
            nonlocal call_count
            if call_count == 0:
                restore_started.set()
                assert allow_restore.wait(timeout=1)
            call_count += 1
            original_batched_memcpy(src_ptrs, dst_ptrs, sizes)

        monkeypatch.setattr(backend, "_batched_memcpy", _blocking_batched_memcpy)

        try:
            _store_tensor(backend, key1, num_tokens=16, fill_value=3)
            _store_tensor(backend, key2, num_tokens=16, fill_value=4)

            def _reader() -> None:
                reader_result.append(backend.batched_get_blocking([key1]))

            def _remover() -> None:
                remove_result.append(backend.remove(key2))
                remove_finished.set()

            reader = threading.Thread(target=_reader)
            reader.start()
            assert restore_started.wait(timeout=1)

            remover = threading.Thread(target=_remover)
            remover.start()
            assert remove_finished.wait(timeout=0.2)
            assert remove_result == [True]

            allow_restore.set()
            reader.join(timeout=1)
            remover.join(timeout=1)
            assert not reader.is_alive()
            assert not remover.is_alive()

            results = reader_result[0]
            assert len(results) == 1
            assert results[0] is not None
            assert torch.all(results[0].tensor == 3)
            results[0].ref_count_down()
        finally:
            backend.close()


def test_dax_backend_remove_during_read_defers_slot_reclaim(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 32 / (1024 * 1024),
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        key = CacheEngineKey("test_model", 1, 0, 703, torch.bfloat16)
        read_started = threading.Event()
        allow_read = threading.Event()
        reader_result: list[MemoryObj | None] = []
        original_do_read = DaxBackend._do_read

        def _blocking_do_read(self, offset, memory_obj, size) -> None:
            read_started.set()
            assert allow_read.wait(timeout=1)
            original_do_read(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_read", _blocking_do_read)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(7)
            backend.batched_submit_put_task([key], [obj])
            obj.ref_count_down()

            def _reader() -> None:
                reader_result.append(backend.get_blocking(key))

            reader = threading.Thread(target=_reader)
            reader.start()
            assert read_started.wait(timeout=1)

            assert backend.remove(key)
            blocked_key = CacheEngineKey("test_model", 1, 0, 704, torch.bfloat16)
            blocked = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert blocked is not None
            with pytest.raises(RuntimeError, match="No free slots available"):
                backend.batched_submit_put_task([blocked_key], [blocked])
            blocked.ref_count_down()

            allow_read.set()
            reader.join(timeout=1)
            assert not reader.is_alive()

            result = reader_result[0]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 7)
            result.ref_count_down()

            recycled_key = CacheEngineKey("test_model", 1, 0, 705, torch.bfloat16)
            recycled = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert recycled is not None
            backend.batched_submit_put_task([recycled_key], [recycled])
            recycled_out = backend.get_blocking(recycled_key)
            assert recycled_out is not None
            recycled_out.ref_count_down()
            recycled.ref_count_down()
            assert backend.get_blocking(key) is None
        finally:
            backend.close()


def test_dax_backend_remove_during_batched_restore_defers_slot_reclaim(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 32 / (1024 * 1024),
                "dax.restore_workers": 1,
                "dax.restore_max_regions": 1,
                "dax.retrieve_staging_slab_bytes": 65536,
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        key = CacheEngineKey("test_model", 1, 0, 713, torch.bfloat16)
        restore_started = threading.Event()
        allow_restore = threading.Event()
        reader_result: list[list[MemoryObj | None]] = []
        original_batched_memcpy = backend._batched_memcpy
        call_count = 0

        def _blocking_batched_memcpy(src_ptrs, dst_ptrs, sizes) -> None:
            nonlocal call_count
            if call_count == 0:
                restore_started.set()
                assert allow_restore.wait(timeout=1)
            call_count += 1
            original_batched_memcpy(src_ptrs, dst_ptrs, sizes)

        monkeypatch.setattr(backend, "_batched_memcpy", _blocking_batched_memcpy)

        try:
            _store_tensor(backend, key, num_tokens=256, fill_value=7)

            def _reader() -> None:
                reader_result.append(backend.batched_get_blocking([key]))

            reader = threading.Thread(target=_reader)
            reader.start()
            assert restore_started.wait(timeout=1)

            assert backend.remove(key)
            blocked_key = CacheEngineKey("test_model", 1, 0, 714, torch.bfloat16)
            blocked = _allocate_kv_obj(num_tokens=256, fill_value=11)
            try:
                with pytest.raises(RuntimeError, match="No free slots available"):
                    backend.batched_submit_put_task([blocked_key], [blocked])
            finally:
                blocked.ref_count_down()

            allow_restore.set()
            reader.join(timeout=1)
            assert not reader.is_alive()

            results = reader_result[0]
            assert len(results) == 1
            assert results[0] is not None
            assert torch.all(results[0].tensor == 7)
            results[0].ref_count_down()

            recycled_key = CacheEngineKey("test_model", 1, 0, 715, torch.bfloat16)
            _store_tensor(backend, recycled_key, num_tokens=256, fill_value=9)
            recycled_results = backend.batched_get_blocking([recycled_key])
            assert recycled_results[0] is not None
            recycled_results[0].ref_count_down()
            assert backend.get_blocking(key) is None
        finally:
            backend.close()


def test_dax_backend_get_read_failure_releases_cpu_memory_obj(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 32 / (1024 * 1024),
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        allocated_on_read: list[MemoryObj] = []
        original_allocate = local_cpu.allocate

        def _tracking_allocate(
            shape: torch.Size,
            dtype: torch.dtype,
            fmt: MemoryFormat,
        ) -> MemoryObj:
            memory_obj = original_allocate(shape, dtype, fmt)
            assert memory_obj is not None
            allocated_on_read.append(memory_obj)
            return memory_obj

        def _failing_do_read(
            self,
            offset: int,
            memory_obj: MemoryObj,
            size: int,
        ) -> None:
            del self, offset, memory_obj, size
            raise RuntimeError("simulated read failure")

        monkeypatch.setattr(local_cpu, "allocate", _tracking_allocate)
        monkeypatch.setattr(DaxBackend, "_do_read", _failing_do_read)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 707, torch.bfloat16)
            obj = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(7)
            backend.batched_submit_put_task([key], [obj])
            obj.ref_count_down()

            with pytest.raises(RuntimeError, match="simulated read failure"):
                backend.get_blocking(key)

            assert len(allocated_on_read) == 1
            assert allocated_on_read[0].get_ref_count() == 0
            assert backend.contains(key)
        finally:
            backend.close()


def test_dax_backend_allocator_exhaustion_fails_without_internal_eviction(
    memory_allocator,
    loop_in_thread,
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 64 / (1024 * 1024),  # ~2 slots for test kv shape
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            keys = [
                CacheEngineKey("test_model", 1, 0, 101, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 102, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 103, torch.bfloat16),
            ]

            for i, key in enumerate(keys[:2]):
                obj = alloc.allocate(
                    [torch.Size([2, 256, 8])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None
                obj.tensor.fill_(i + 1)
                futs = backend.batched_submit_put_task([key], [obj])
                if futs:
                    for fut in futs:
                        fut.result(timeout=5)
                obj.ref_count_down()

            obj3 = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj3 is not None
            with pytest.raises(RuntimeError, match="No free slots available"):
                backend.batched_submit_put_task([keys[2]], [obj3])
            obj3.ref_count_down()

            out0 = backend.get_blocking(keys[0])
            assert out0 is not None
            out0.ref_count_down()
            out1 = backend.get_blocking(keys[1])
            assert out1 is not None
            out1.ref_count_down()
            assert backend.get_blocking(keys[2]) is None
        finally:
            backend.close()


def test_dax_backend_full_arena_does_not_evict_pinned_or_unpinned_keys(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 64 / (1024 * 1024),
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            keys = [
                CacheEngineKey("test_model", 1, 0, 201, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 202, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 203, torch.bfloat16),
            ]

            for key in keys[:2]:
                obj = alloc.allocate(
                    [torch.Size([2, 256, 8])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                futs = backend.batched_submit_put_task([key], [obj])
                if futs:
                    for fut in futs:
                        fut.result(timeout=5)
                obj.ref_count_down()

            assert backend.pin(keys[0])

            obj3 = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj3 is not None
            with pytest.raises(RuntimeError, match="No free slots available"):
                futs = backend.batched_submit_put_task([keys[2]], [obj3])
                if futs:
                    for fut in futs:
                        fut.result(timeout=5)
            obj3.ref_count_down()

            out0 = backend.get_blocking(keys[0])
            assert out0 is not None
            out0.ref_count_down()
            out1 = backend.get_blocking(keys[1])
            assert out1 is not None
            out1.ref_count_down()
            assert backend.get_blocking(keys[2]) is None
        finally:
            backend.close()


def test_dax_backend_overlapping_pin_unpin(memory_allocator, loop_in_thread):
    """Regression: overlapping pin/unpin must use ref-counting, not a plain set."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 64 / (1024 * 1024),  # 2 slots
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            keys = [
                CacheEngineKey("test_model", 1, 0, 401, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 402, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 403, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 404, torch.bfloat16),
            ]

            # Store keys A and B (fills both slots)
            for key in keys[:2]:
                obj = alloc.allocate(
                    [torch.Size([2, 256, 8])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                futs = backend.batched_submit_put_task([key], [obj])
                if futs:
                    for fut in futs:
                        fut.result(timeout=5)
                obj.ref_count_down()

            # Pin key A twice (simulate two concurrent lookups)
            assert backend.pin(keys[0])
            assert backend.pin(keys[0])

            # Unpin key A once — pin_count should still be 1
            backend.unpin(keys[0])

            # Key A still has one outstanding pin and cannot be removed
            # by non-forced eviction-controller delete calls.
            assert not backend.remove(keys[0], force=False)
            assert backend.remove(keys[1], force=False)

            obj_c = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj_c is not None
            futs = backend.batched_submit_put_task([keys[2]], [obj_c])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)
            obj_c.ref_count_down()

            assert backend.contains(keys[0]), (
                "key A should survive non-forced delete while pin_count=1"
            )
            assert not backend.contains(keys[1]), "key B should be deleted"
            assert backend.contains(keys[2])

            # Unpin key A a second time; non-forced delete can now reclaim it.
            backend.unpin(keys[0])
            assert backend.remove(keys[0], force=False)

            obj_d = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj_d is not None
            futs = backend.batched_submit_put_task([keys[3]], [obj_d])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)
            obj_d.ref_count_down()

            assert not backend.contains(keys[0]), (
                "key A should be removable after full unpin"
            )
            assert backend.contains(keys[3])
        finally:
            backend.close()


def test_dax_backend_remove_inflight_reclaims_slot(memory_allocator, loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 32 / (1024 * 1024),  # one slot
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            key1 = CacheEngineKey("test_model", 1, 0, 301, torch.bfloat16)
            obj1 = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj1 is not None
            backend.batched_submit_put_task([key1], [obj1])
            obj1.ref_count_down()

            assert backend.remove(key1)
            assert backend.get_blocking(key1) is None

            key2 = CacheEngineKey("test_model", 1, 0, 302, torch.bfloat16)
            obj2 = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj2 is not None
            backend.batched_submit_put_task([key2], [obj2])
            obj2.ref_count_down()
            out = backend.get_blocking(key2)
            assert out is not None
            out.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_multithread_put_get_smoke(memory_allocator, loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        def _worker(i: int) -> None:
            alloc = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 400 + i, torch.bfloat16)
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(i)
            futs = backend.batched_submit_put_task([key], [obj])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)
            obj.ref_count_down()
            out = backend.get_blocking(key)
            assert out is not None
            assert out.tensor is not None
            assert torch.equal(out.tensor, obj.tensor)
            out.ref_count_down()

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_worker, i) for i in range(20)]
                for fut in futures:
                    fut.result(timeout=10)
        finally:
            backend.close()


def test_dax_backend_sync_close_waits_for_active_put(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        write_started = threading.Event()
        allow_write = threading.Event()
        close_returned = threading.Event()
        observed: dict[str, object] = {}
        writer_exc: dict[str, BaseException] = {}
        original_do_write = DaxBackend._do_write

        def _blocking_do_write(self, offset, memory_obj, size) -> None:
            write_started.set()
            assert allow_write.wait(timeout=1)
            observed["mmap_is_none"] = self._mmap_obj is None
            observed["base_ptr"] = self._base_ptr
            original_do_write(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_write", _blocking_do_write)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            key = CacheEngineKey("test_model", 1, 0, 410, torch.bfloat16)

            def _writer() -> None:
                try:
                    backend.batched_submit_put_task([key], [obj])
                except BaseException as e:
                    writer_exc["error"] = e

            def _closer() -> None:
                backend.close()
                close_returned.set()

            writer = threading.Thread(target=_writer)
            closer = threading.Thread(target=_closer)

            writer.start()
            assert write_started.wait(timeout=1)
            closer.start()
            time.sleep(0.05)
            assert not close_returned.is_set()

            allow_write.set()
            writer.join(timeout=1)
            closer.join(timeout=1)

            assert not writer.is_alive()
            assert not closer.is_alive()
            assert "error" not in writer_exc
            assert close_returned.is_set()
            assert observed["mmap_is_none"] is False
            assert observed["base_ptr"] != 0
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_sync_close_waits_for_active_get(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        read_started = threading.Event()
        allow_read = threading.Event()
        close_returned = threading.Event()
        reader_result: list[MemoryObj | None] = []
        original_do_read = DaxBackend._do_read

        def _blocking_do_read(self, offset, memory_obj, size) -> None:
            read_started.set()
            assert allow_read.wait(timeout=1)
            original_do_read(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_read", _blocking_do_read)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(6)
            key = CacheEngineKey("test_model", 1, 0, 412, torch.bfloat16)
            backend.batched_submit_put_task([key], [obj])
            obj.ref_count_down()

            def _reader() -> None:
                reader_result.append(backend.get_blocking(key))

            def _closer() -> None:
                backend.close()
                close_returned.set()

            reader = threading.Thread(target=_reader)
            closer = threading.Thread(target=_closer)

            reader.start()
            assert read_started.wait(timeout=1)
            closer.start()
            time.sleep(0.05)

            assert not close_returned.is_set()
            assert backend._mmap_obj is not None  # noqa: SLF001
            assert backend._base_ptr != 0  # noqa: SLF001

            allow_read.set()
            reader.join(timeout=1)
            closer.join(timeout=1)

            assert not reader.is_alive()
            assert not closer.is_alive()
            assert close_returned.is_set()

            result = reader_result[0]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 6)
            result.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_sync_close_waits_for_active_batched_restore(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        restore_started = threading.Event()
        allow_restore = threading.Event()
        close_returned = threading.Event()
        reader_result: list[list[MemoryObj | None]] = []
        original_batched_memcpy = backend._batched_memcpy
        call_count = 0

        def _blocking_batched_memcpy(src_ptrs, dst_ptrs, sizes) -> None:
            nonlocal call_count
            if call_count == 0:
                restore_started.set()
                assert allow_restore.wait(timeout=1)
            call_count += 1
            original_batched_memcpy(src_ptrs, dst_ptrs, sizes)

        monkeypatch.setattr(backend, "_batched_memcpy", _blocking_batched_memcpy)

        key = CacheEngineKey("test_model", 1, 0, 719, torch.bfloat16)

        try:
            _store_tensor(backend, key, num_tokens=16, fill_value=6)

            def _reader() -> None:
                reader_result.append(backend.batched_get_blocking([key]))

            reader = threading.Thread(target=_reader)
            reader.start()
            assert restore_started.wait(timeout=1)

            def _closer() -> None:
                backend.close()
                close_returned.set()

            closer = threading.Thread(target=_closer)
            closer.start()

            time.sleep(0.1)
            assert not close_returned.is_set()

            allow_restore.set()
            reader.join(timeout=1)
            closer.join(timeout=1)
            assert close_returned.is_set()
            assert not reader.is_alive()
            assert not closer.is_alive()

            results = reader_result[0]
            assert len(results) == 1
            assert results[0] is not None
            results[0].ref_count_down()
        finally:
            backend.close()


def test_dax_backend_close_rejects_new_ops_after_shutdown() -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=AdHocMemoryAllocator(device="cpu"),
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=None,
            dst_device="cpu",
        )

        try:
            backend.close()
            assert (
                backend.get_blocking(
                    CacheEngineKey("test_model", 1, 0, 999, torch.bfloat16)
                )
                is None
            )
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            with pytest.raises(RuntimeError, match="closing"):
                backend.batched_submit_put_task(
                    [CacheEngineKey("test_model", 1, 0, 1000, torch.bfloat16)],
                    [obj],
                )
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_async_close_waits_for_active_put(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 16 / 1024,
                "dax.async_put": True,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        write_started = threading.Event()
        allow_write = threading.Event()
        close_returned = threading.Event()
        original_do_write = DaxBackend._do_write

        def _blocking_do_write(self, offset, memory_obj, size) -> None:
            write_started.set()
            assert allow_write.wait(timeout=2)
            original_do_write(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_write", _blocking_do_write)

        # Start the event loop so the async_put path is exercised.
        loop_thread = threading.Thread(target=loop_in_thread.run_forever, daemon=True)
        loop_thread.start()
        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 411, torch.bfloat16)
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None

            def _closer() -> None:
                backend.close()
                close_returned.set()

            assert write_started.wait(timeout=2)
            closer = threading.Thread(target=_closer)
            closer.start()
            time.sleep(0.05)
            assert not close_returned.is_set()

            allow_write.set()
            closer.join(timeout=2)

            assert not closer.is_alive()
            assert close_returned.is_set()
            obj.ref_count_down()
        finally:
            loop_in_thread.call_soon_threadsafe(loop_in_thread.stop)
            loop_thread.join(timeout=2)
            backend.close()
