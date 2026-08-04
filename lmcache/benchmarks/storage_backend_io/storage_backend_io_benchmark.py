# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
#
# 2026/4/20 support benchmark write performance of hf3fs and fs backend
#   Wenwen Chen <wenwen.chen@samsung.com>
#   Ruyi Zhang <ruyi.zhang@samsung.com>

"""Benchmark storage backends under high write/read concurrency.
This module provides a framework for benchmarking different storage backends
(LocalDiskBackend, RustRawBlockBackend, RemoteBackend, etc.) with consistent
logic. Supports both write and read benchmarks with optional integrity verification.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional
import argparse
import asyncio
import json
import os
import stat
import tempfile
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

logger = init_logger(__name__)
# Type aliases
OnCompleteCallback = Callable[[CacheEngineKey], None]


# ============================================================================
# Constants
# ============================================================================
DEFAULT_CHUNK_SIZE = 256

# 2nd dim should equal to chunk size
DEFAULT_KV_SHAPE = (28, 2, 256, 8, 128)
# DEFAULT_SHAPE_LIST = [2, 28, 256, 1024]
DEFAULT_SHAPE_LIST = [2, 16, 256, 128]

DEFAULT_TORCH_SHAPE = torch.Size(DEFAULT_SHAPE_LIST)
DEFAULT_DTYPE = torch.bfloat16  # 2Bytes

# ============================================================================
# Helper Functions
# ============================================================================


def _start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start an async event loop in a background thread."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="bench-loop", daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop: asyncio.AbstractEventLoop, t: threading.Thread) -> None:
    """Stop the async event loop."""
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


def _build_metadata(chunk_size: int) -> LMCacheMetadata:
    """Build test metadata for benchmark."""
    kv_shape = (
        DEFAULT_KV_SHAPE[0],
        DEFAULT_KV_SHAPE[1],
        chunk_size,
        DEFAULT_KV_SHAPE[3],
        DEFAULT_KV_SHAPE[4],
    )
    logger.info(f"_build_metadata: chunk_size {chunk_size}, kv_shape {kv_shape}")
    return LMCacheMetadata(
        model_name="benchmark_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=kv_shape,
        chunk_size=chunk_size,
    )


def _make_memory_objs(
    num_ops: int,
    use_aligned: bool,
    alignment: int,
    keepalive: list[torch.Tensor],
    memory_allocator: Optional[MemoryAllocatorInterface] = None,
    shapes: Optional[list[torch.Size]] = None,
    start_idx: int = 0,
) -> list:
    """Create memory objects for benchmark."""
    if memory_allocator is None:
        memory_allocator = AdHocMemoryAllocator(device="cpu")
    if shapes is None:
        shapes = [DEFAULT_TORCH_SHAPE]
    logger.info(f"_make_memory_objs, shapes: {shapes}")

    objs = []
    for i in range(num_ops):
        if use_aligned:
            num_bytes = shapes[0].numel() * DEFAULT_DTYPE.itemsize
            base = torch.empty(
                torch.Size([num_bytes + alignment]),
                dtype=torch.uint8,
                device="cpu",
            )
            offset = (-base.data_ptr()) % alignment
            aligned = base[offset : offset + num_bytes]
            keepalive.append(base)
            obj = TensorMemoryObj(
                raw_data=aligned,
                metadata=MemoryObjMetadata(
                    shape=shapes[0],
                    dtype=DEFAULT_DTYPE,
                    address=0,
                    phy_size=0,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.KV_T2D,
                    shapes=shapes,
                    dtypes=[DEFAULT_DTYPE],
                ),
                parent_allocator=memory_allocator,
            )
        else:
            allocated_obj = memory_allocator.allocate(
                shapes,
                [DEFAULT_DTYPE],
                fmt=MemoryFormat.KV_T2D,
            )
            assert allocated_obj is not None
            obj = allocated_obj  # type: ignore[assignment]
        assert obj.tensor is not None
        obj.tensor.fill_(start_idx + i)
        objs.append(obj)
    return objs


def _release_memory_objs(objs: list) -> None:
    """Release memory objects."""
    for obj in objs:
        try:
            obj.ref_count_down()
        except Exception:
            pass


def _make_keys(num_ops: int) -> list[CacheEngineKey]:
    """Create cache keys for benchmark."""
    return [
        CacheEngineKey("benchmark_model", 1, 0, i, DEFAULT_DTYPE)
        for i in range(num_ops)
    ]


def _compute_required_buffer_gb(
    metadata: LMCacheMetadata,
    num_ops: int,
    chunk_size: int,
) -> float:
    """Compute the required CPU buffer size in GB for the benchmark.

    Calculates the total size of all chunks based on KV shape and dtype,
    then converts to GB with a small floor margin.

    Args:
        metadata: The benchmark metadata containing KV shape and dtype info.
        num_ops: Number of put/get operations.
        chunk_size: The chunk size used by the backend.

    Returns:
        Required buffer size in GB (minimum 0.01 GB).
    """
    chunk_size_bytes = (
        metadata.kv_shape[0]
        * metadata.kv_shape[1]
        * chunk_size
        * metadata.kv_shape[3]
        * metadata.kv_shape[4]
        * metadata.kv_dtype.itemsize
    )
    return max(0.01, (num_ops * chunk_size_bytes) / (1024**3))


# ============================================================================
# Abstract Base Class for Storage Backends
# ============================================================================
class StorageBackendBenchmark(ABC):
    """Abstract base class for storage backend benchmarks.

    This class provides common benchmark logic and defines abstract methods
    that each backend implementation must override.
    """

    def __init__(
        self,
        name: str,
        num_ops: int,
        concurrency: int,
        use_odirect: bool,
        alignment: int,
        write_bench: bool,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verify_integrity: bool = False,
        use_uring: bool = False,
    ):
        op = "write_" if write_bench else "read_"
        self._backend_name = op + name
        self.num_ops = num_ops
        self.concurrency = concurrency
        self.use_odirect = use_odirect
        self.alignment = alignment
        self.write_bench = write_bench
        self.chunk_size = chunk_size
        self.verify_integrity = verify_integrity
        self.use_uring = use_uring

        # Runtime state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._local_cpu: LocalCPUBackend
        self._backend: StorageBackendInterface
        self._keys: list[CacheEngineKey] = []
        self._objs: list[TensorMemoryObj] = []
        self._keepalive: list[torch.Tensor] = []
        self._start_time: float
        # completed ops
        self._completed = 0
        # lock for _completed
        self._lock = threading.Lock()
        self._done = threading.Event()
        # Flag to skip cleanup in run() if already handled in _execute_benchmark()
        self._skip_cleanup = False

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    @abstractmethod
    def extra_config_keys(self) -> dict:
        """Return extra config keys specific to this backend."""
        pass

    @abstractmethod
    def _create_backend(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> StorageBackendInterface:
        """Create and return the backend instance."""
        pass

    @abstractmethod
    def _close_backend(self) -> None:
        """Close the backend and perform cleanup."""
        pass

    def _submit_tasks(
        self,
        keys: list[CacheEngineKey],
        objs: Optional[list] = None,
    ) -> list[Any]:
        """Submit tasks (put or get) based on the benchmark mode.

        Args:
            keys: List of cache keys.
            objs: List of memory objects for put operations. If None, submit get tasks.

        Returns:
            List of pending operations (futures or empty list).
        """
        # Default implementation: submit put tasks
        return self._submit_put_tasks(keys, objs if objs else [])

    def _submit_put_tasks(
        self,
        keys: list[CacheEngineKey],
        objs: list,
    ) -> list[Any]:
        """Submit put tasks using callback pattern."""
        self._completed = 0
        self._done.clear()

        def on_complete(_key: CacheEngineKey) -> None:
            with self._lock:
                self._completed += 1
                if self._completed >= self.num_ops:
                    self._done.set()

        def submit_slice(start: int, end: int) -> None:
            self._backend.batched_submit_put_task(
                keys[start:end],
                objs[start:end],
                on_complete_callback=on_complete,
            )

        slice_size = max(1, self.num_ops // self.concurrency)
        slices = [
            (i, min(i + slice_size, self.num_ops))
            for i in range(0, self.num_ops, slice_size)
        ]

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for s in slices:
                ex.submit(submit_slice, s[0], s[1])

        # Return empty list since we use callback pattern
        return []

    def _wait_for_completion(self, pending_ops: list[Any]) -> None:
        """Wait using Event pattern."""
        # Keep a floor for normal runs but scale for large-op runs.
        # This avoids premature timeout for long single-shot benchmarks.
        timeout_sec = max(300.0, float(self.num_ops) / 100.0)
        while not self._done.wait(timeout=1.0):
            if self._completed >= self.num_ops:
                break
            if float(time.perf_counter() - self._start_time) >= timeout_sec:
                raise TimeoutError(
                    f"{self.backend_name} benchmark timed out: "
                    f"completed={self._completed}, expected={self.num_ops}"
                )

    def _setup_device(self) -> None:
        return None

    def _cleanup_device(self) -> None:
        return None

    def _setup_config(self, base_config: LMCacheEngineConfig) -> LMCacheEngineConfig:
        """Setup backend-specific configuration. Override if needed."""
        return base_config

    def run(self) -> dict:
        """Run the benchmark and return results."""
        # Setup
        self._loop, self._loop_thread = _start_loop()
        metadata = _build_metadata(self.chunk_size)
        logger.info(f"Prepare config for {self.backend_name} ...")

        rust_raw = "rust" in self.backend_name

        required_buffer_gb = (
            _compute_required_buffer_gb(metadata, self.num_ops, self.chunk_size)
            if rust_raw
            else 0.1
        )

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=self.chunk_size,
            local_cpu=True,
            max_local_cpu_size=required_buffer_gb,
            lmcache_instance_id=f"bench_{self.backend_name}",
        )

        # prepare the raw disk
        self._setup_device()
        config.extra_config = self.extra_config_keys
        config = self._setup_config(config)

        # Create local CPU backend (common to all backends)
        if self.use_uring and rust_raw:
            self._local_cpu = LocalCPUBackend(
                config=config,
                metadata=metadata,
                dst_device="cpu",
            )
        else:
            self._local_cpu = LocalCPUBackend(
                config=config,
                metadata=metadata,
                dst_device="cpu",
                memory_allocator=AdHocMemoryAllocator(device="cpu"),
            )
        logger.info(f"Creating {self.backend_name} ...")
        # Create the specific backend
        self._backend = self._create_backend(
            config, metadata, self._loop, self._local_cpu
        )

        # Prepare test data
        self._keys = _make_keys(self.num_ops)
        shapes = metadata.get_shapes()
        self._objs = _make_memory_objs(
            self.num_ops,
            self.use_odirect,
            self.alignment,
            self._keepalive,
            memory_allocator=self._local_cpu.memory_allocator,
            shapes=shapes,  # [torch.Size(shape)]
        )

        # Run benchmark
        logger.info(f"Start benchmark with {self.backend_name} ...")
        self._start_time = time.perf_counter()
        result = self._execute_benchmark()
        logger.info(f"End benchmark with {self.backend_name} ...")
        # Cleanup (skip if already handled in _execute_benchmark)
        if not self._skip_cleanup:
            _release_memory_objs(self._objs)
        self._close_backend()
        logger.info(f"Closed {self.backend_name} ...")

        self._cleanup_device()
        _stop_loop(self._loop, self._loop_thread)
        return result

    def _get_slices(self) -> list[tuple[int, int]]:
        """Get slices for parallel task submission."""
        slice_size = max(1, self.num_ops // self.concurrency)
        return [
            (i, min(i + slice_size, self.num_ops))
            for i in range(0, self.num_ops, slice_size)
        ]

    def _execute_write_phase(self) -> float:
        """Execute the write phase and return elapsed time in seconds.

        This is shared between single-phase (write-only) and two-phase
        (write+read) benchmarks to avoid code duplication.
        """
        start = time.perf_counter()
        pending_ops = self._submit_tasks(self._keys, self._objs) or []

        if pending_ops or self._uses_futures_pattern():
            self._wait_for_futures(pending_ops)
        else:
            self._wait_for_completion(pending_ops)

        return time.perf_counter() - start

    def _execute_benchmark(self) -> dict:
        """Execute the benchmark with concurrent writes.

        Supports two modes:
        - write_bench=True: Single-phase write-only benchmark
        - write_bench=False: Two-phase write+read benchmark (if supported by backend)
        """
        # Check if this is a two-phase benchmark (write + read)
        if not self.write_bench:
            return self._execute_two_phase_benchmark()

        # Single-phase benchmark (write-only)
        elapsed = self._execute_write_phase()

        return {
            "backend": self.backend_name,
            "num_ops": self.num_ops,
            "concurrency": self.concurrency,
            "write_elapsed_sec": elapsed,
            "write_ops_per_sec": self.num_ops / elapsed if elapsed > 0 else 0.0,
            "use_odirect": self.use_odirect,
        }

    def _uses_futures_pattern(self) -> bool:
        """Check if this backend uses futures pattern.

        Override in subclass to indicate futures pattern usage.
        """
        return False

    def _wait_for_futures(self, futures: list[Any]) -> None:
        """Wait for all futures to complete."""
        for fut in futures:
            fut.result(timeout=120)

    def _get_extra_result_fields(self) -> dict:
        """Get extra fields to add to the benchmark result.

        Override in subclass to add backend-specific fields.
        """
        return {}

    def _execute_two_phase_benchmark(self) -> dict:
        """Execute two-phase benchmark: write then read.

        This method handles the write phase, release of write objects,
        read phase, and optional integrity verification.
        Subclasses should override _verify_integrity() if needed.
        """
        # Store original data for integrity verification
        original_data: dict[CacheEngineKey, torch.Tensor] = {}
        if self.verify_integrity:
            for key, obj in zip(self._keys, self._objs, strict=False):
                assert obj.tensor is not None
                original_data[key] = obj.tensor.clone()

        # Write phase
        logger.info("Begin to test write performance")
        write_elapsed = self._execute_write_phase()
        logger.info("End of test write performance")
        # Release write objects
        _release_memory_objs(self._objs)
        self._skip_cleanup = True

        # Read phase
        logger.info("Begin to test read performance")
        read_start = time.perf_counter()
        read_results = self._execute_read_phase()
        read_elapsed = time.perf_counter() - read_start
        logger.info("End of test read performance")

        # Verify integrity
        integrity_errors = 0
        integrity_passed = False
        if self.verify_integrity:
            integrity_errors, integrity_passed = self._verify_integrity(
                read_results, original_data
            )
            logger.info("End of verify integrity")

        # Cleanup read objects
        self._cleanup_read_results(read_results)

        result = {
            "backend": self.backend_name,
            "num_ops": self.num_ops,
            "concurrency": self.concurrency,
            "write_elapsed_sec": write_elapsed,
            "write_ops_per_sec": self.num_ops / write_elapsed
            if write_elapsed > 0
            else 0.0,
            "read_elapsed_sec": read_elapsed,
            "read_ops_per_sec": self.num_ops / read_elapsed
            if read_elapsed > 0
            else 0.0,
            "total_elapsed_sec": write_elapsed + read_elapsed,
            "use_odirect": self.use_odirect,
            "verify_integrity": self.verify_integrity,
            "integrity_errors": integrity_errors,
            "integrity_passed": integrity_passed,
        }
        # Add extra fields from subclass
        result.update(self._get_extra_result_fields())
        return result

    def _execute_read_phase(self) -> list[tuple[CacheEngineKey, Optional[MemoryObj]]]:
        """Execute the read phase and return results.

        Override in subclass to customize read behavior.
        """
        slices = self._get_slices()
        read_results: list[tuple[CacheEngineKey, Optional[MemoryObj]]] = []
        read_lock = threading.Lock()

        def submit_read_slice(start: int, end: int) -> None:
            batch_keys = self._keys[start:end]
            loaded = self._backend.batched_get_blocking(batch_keys)
            with read_lock:
                read_results.extend(zip(batch_keys, loaded, strict=False))

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for s in slices:
                ex.submit(submit_read_slice, s[0], s[1])

        return read_results

    def _verify_integrity(
        self,
        read_results: list[tuple[CacheEngineKey, Optional[MemoryObj]]],
        original_data: dict[CacheEngineKey, torch.Tensor],
    ) -> tuple[int, bool]:
        """Verify data integrity between original and read data.

        Returns:
            Tuple of (error_count, is_passed)
        """
        errors = 0
        for key, read_obj in read_results:
            if read_obj is None:
                errors += 1
                continue
            assert read_obj.tensor is not None
            if key not in original_data:
                errors += 1
                continue
            if not torch.equal(read_obj.tensor, original_data[key]):
                errors += 1
        return errors, errors == 0

    def _cleanup_read_results(
        self,
        read_results: list[tuple[CacheEngineKey, Optional[MemoryObj]]],
    ) -> None:
        """Cleanup read results by releasing memory objects."""
        for _, read_obj in read_results:
            if read_obj is not None:
                try:
                    read_obj.ref_count_down()
                except Exception:
                    pass


# ============================================================================
# LocalDiskBackend Implementation
# ============================================================================
class LocalDiskBackendBenchmark(StorageBackendBenchmark):
    """Benchmark for LocalDiskBackend."""

    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        local_disk_dir: str,
        max_disk_gb: float,
        use_odirect: bool,
        alignment: int,
        write_bench: bool,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verify_integrity: bool = False,
    ):
        super().__init__(
            "local_disk",
            num_ops,
            concurrency,
            use_odirect,
            alignment,
            write_bench,
            chunk_size,
            verify_integrity,
        )
        self.local_disk_dir = local_disk_dir
        self.max_disk_gb = max_disk_gb

    @property
    def extra_config_keys(self) -> dict:
        return {"use_odirect": self.use_odirect}

    def _setup_config(self, config: LMCacheEngineConfig) -> LMCacheEngineConfig:
        config.local_disk = self.local_disk_dir
        config.max_local_disk_size = self.max_disk_gb
        return config

    def _create_backend(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> LocalDiskBackend:
        return LocalDiskBackend(
            config=config,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )

    def _close_backend(self) -> None:
        self._backend.close()


# ============================================================================
# RustRawBlockBackend Implementation
# ============================================================================
class RustRawBlockBackendBenchmark(StorageBackendBenchmark):
    """Benchmark for RustRawBlockBackend.

    Supports both write and read benchmarks:
    - write_bench=True: Perform write-only benchmark
    - write_bench=False: Perform write+read benchmark with optional
    - integrity verification
    """

    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        raw_device: str,
        raw_device_size_gb: float,
        use_odirect: bool,
        alignment: int,
        cleanup_raw_device: bool,
        write_bench: bool,
        use_uring: bool,
        use_uring_cmd: bool,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verify_integrity: bool = False,
        max_data_transfer_size: int = 0,
    ):
        super().__init__(
            "rust_raw_block",
            num_ops,
            concurrency,
            use_odirect,
            alignment,
            write_bench,
            chunk_size,
            verify_integrity,
            use_uring,
        )
        self.raw_device = raw_device
        self.raw_device_size_gb = raw_device_size_gb
        self.cleanup_raw_device = cleanup_raw_device
        self._temp_dir: Optional[str] = None
        self._manifest_path: Optional[str] = None
        self.use_uring = use_uring
        self.use_uring_cmd = use_uring_cmd
        self.max_data_transfer_size = max_data_transfer_size
        # Create manifest path
        self._manifest_path = os.path.join(
            tempfile.gettempdir(),
            f"lmcache_rust_raw_block_bench_{os.getpid()}_{time.time_ns()}.manifest.json",
        )

    @property
    def extra_config_keys(self) -> dict:
        return {
            "rust_raw_block.device_path": self.raw_device,
            "rust_raw_block.block_align": self.alignment,
            "rust_raw_block.header_bytes": self.alignment,
            "rust_raw_block.use_odirect": self.use_odirect,
            "rust_raw_block.manifest_path": self._manifest_path,
            "rust_raw_block.manifest_write_interval": 0,
            "rust_raw_block.use_uring": self.use_uring,
            "rust_raw_block.use_uring_cmd": self.use_uring_cmd,
            "rust_raw_block.max_data_transfer_size": self.max_data_transfer_size,
        }

    def _setup_device(self) -> None:
        """Setup raw block device or temp file."""
        is_block_device = False
        is_char_device = False
        self._temp_dir = None
        prefix = "write" if self.write_bench else "read"
        prefix = prefix + "raw_block_bench_"
        if self.raw_device:
            try:
                st_mode = os.stat(self.raw_device).st_mode
                is_block_device = stat.S_ISBLK(st_mode)
                is_char_device = stat.S_ISCHR(st_mode)
            except FileNotFoundError:
                is_block_device = False
                is_char_device = False

        # Create temp file if no device specified
        if not self.raw_device:
            self._temp_dir = tempfile.mkdtemp(prefix)
            self.raw_device = os.path.join(self._temp_dir, "raw_block.bin")

        # Truncate if not a real block device or character device
        if self.raw_device and not is_block_device and not is_char_device:
            with open(self.raw_device, "wb") as f:
                f.truncate(int(self.raw_device_size_gb * 1024**3))

    def _create_backend(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> RustRawBlockBackend:
        return RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
            dst_device="cpu",
        )

    def _uses_futures_pattern(self) -> bool:
        """RustRawBlockBackend uses futures pattern."""
        return True

    def _submit_put_tasks(
        self,
        keys: list[CacheEngineKey],
        objs: list,
    ) -> list[Any]:
        """Submit put tasks using futures pattern (for RustRawBlockBackend)."""
        futures: list[Future] = []
        fut_lock = threading.Lock()
        slices = self._get_slices()

        def submit_slice(start: int, end: int) -> None:
            result = self._backend.batched_submit_put_task(
                keys[start:end], objs[start:end]
            )
            if result is not None:
                with fut_lock:
                    futures.extend(result)

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for s in slices:
                ex.submit(submit_slice, s[0], s[1])

        return futures

    def _close_backend(self) -> None:
        if self._manifest_path:
            try:
                os.remove(self._manifest_path)
            except Exception:
                pass
        if self._backend:
            self._backend.close()

    def _get_extra_result_fields(self) -> dict:
        """Get extra fields for RustRawBlockBackend benchmark results."""
        return {
            "use_uring": self.use_uring,
            "use_uring_cmd": self.use_uring_cmd,
        }

    def _cleanup_device(self) -> None:
        """Cleanup temp files."""
        if self.cleanup_raw_device or self._temp_dir:
            try:
                os.remove(self.raw_device)
            except Exception:
                pass
            if self._temp_dir:
                try:
                    os.rmdir(self._temp_dir)
                except Exception:
                    pass


# ============================================================================
# RemoteBackendBenchmark Implementation
# ============================================================================
class RemoteBackendBenchmark(StorageBackendBenchmark):
    """Benchmark for RemoteBackend (e.g., S3, Redis)."""

    def __init__(
        self,
        name: str,
        num_ops: int,
        concurrency: int,
        remote_url: str,
        use_odirect: bool,
        alignment: int,
        write_bench: bool,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verify_integrity: bool = False,
    ):
        super().__init__(
            name,
            num_ops,
            concurrency,
            use_odirect,
            alignment,
            write_bench,
            chunk_size,
            verify_integrity,
        )
        self.remote_url = remote_url

    def _setup_config(self, config: LMCacheEngineConfig) -> LMCacheEngineConfig:
        config.remote_url = self.remote_url
        return config

    def _create_backend(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> RemoteBackend:
        return RemoteBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )

    def _close_backend(self):
        if self._backend:
            self._backend.close()


# ============================================================================
# Hf3fsBackendBenchmark Implementation
# ============================================================================
class Hf3fsBackendBenchmark(RemoteBackendBenchmark):
    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        remote_url: str,
        use_odirect: bool,
        alignment: int,
        write_bench: bool,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verify_integrity: bool = False,
    ):
        super().__init__(
            "hf3fs_backend",
            num_ops,
            concurrency,
            remote_url,
            use_odirect,
            alignment,
            write_bench,
            chunk_size,
            verify_integrity,
        )

    @property
    def extra_config_keys(self) -> dict:
        return {
            "hf3fs_mount_point": "/3fs/stage",
            "hf3fs_iov_size": 209715200,
            "hf3fs_ior_entries": 256,
            "hf3fs_io_depth": 0,
            "hf3fs_numa_id": -1,
            "hf3fs_io_thread_num": 8,
        }


# ============================================================================
# FsBackendBenchmark Implementation
# ============================================================================
class FsBackendBenchmark(RemoteBackendBenchmark):
    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        remote_url: str,
        use_odirect: bool,
        alignment: int,
        write_bench: bool,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verify_integrity: bool = False,
    ):
        super().__init__(
            "fs_backend",
            num_ops,
            concurrency,
            remote_url,
            use_odirect,
            alignment,
            write_bench,
            chunk_size,
            verify_integrity,
        )

    @property
    def extra_config_keys(self) -> dict:
        return {
            "save_chunk_meta": False,
            "fs_connector_read_ahead_size": 0,
            "fs_connector_use_odirect": False,
            # "fs_connector_relative_tmp_dir": "tmp",
        }


# ============================================================================
# BigtableBackendBenchmark Implementation
# ============================================================================
class BigtableBackendBenchmark(RemoteBackendBenchmark):
    def __init__(
        self,
        num_ops: int,
        concurrency: int,
        remote_url: str,
        use_odirect: bool,
        alignment: int,
        write_bench: bool,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verify_integrity: bool = False,
    ):
        super().__init__(
            "bigtable",
            num_ops,
            concurrency,
            remote_url,
            use_odirect,
            alignment,
            write_bench,
            chunk_size,
            verify_integrity,
        )

    @property
    def extra_config_keys(self) -> dict:
        return {}


# Main Entry Point
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark storage backends under high write/read concurrency."
    )
    parser.add_argument("--num-ops", type=int, default=256, help="Total put ops")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Number of submit threads"
    )
    parser.add_argument(
        "--backend",
        choices=[
            "local_disk",
            "rust_raw_block",
            "both",
            "hf3fs_backend",
            "fs_backend",
            "bigtable",
        ],
        default="both",
        help=(
            "Backend to benchmark."
            "For both, it will benchmark local_disk and rust_raw_block"
        ),
    )
    parser.add_argument(
        "--write_bench",
        type=str,
        default="True",
        help="Perform write benchmark (default: True). False for read benchmark.",
    )
    parser.add_argument(
        "--use-uring",
        action="store_true",
        help="Enable io_uring for raw block backend",
    )
    parser.add_argument(
        "--use-uring-cmd",
        action="store_true",
        help=(
            "Enable io_uring_cmd for raw block backend. "
            "This automatically enables --use-uring. "
            "Must use nvme character device node (/dev/ngXnY)"
        ),
    )
    parser.add_argument(
        "--max-data-transfer-size",
        type=int,
        default=0,
        help=(
            "Maximum data transfer size for use_uring_cmd. "
            " > 0: Split based on the specified limit. "
            " <= 0: Split based on device reported max hardware sector size"
        ),
    )
    parser.add_argument(
        "--verify-integrity",
        action="store_true",
        help="Verify data integrity after reads (write_bench is False)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for the backend (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--local-disk-dir",
        type=str,
        default="/tmp/lmcache_local_disk_bench",
    )

    parser.add_argument("--max-local-disk-gb", type=float, default=2.0)
    parser.add_argument(
        "--local-disk-odirect",
        action="store_true",
        help="Enable O_DIRECT for local disk backend",
    )
    parser.add_argument(
        "--raw-device",
        type=str,
        default="",
        help="Raw block device path (if empty, uses a temp file)",
    )
    parser.add_argument("--raw-device-size-gb", type=float, default=1.0)
    parser.add_argument(
        "--raw-odirect",
        action="store_true",
        help="Enable O_DIRECT for raw block backend",
    )
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument(
        "--remote-url",
        type=str,
        default="hf3fs:///3fs/stage/hello,/3fs/stage/world",
        help="The url for fs_backend or hf3fs_backend.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON file path or directory",
    )

    args = parser.parse_args()

    # use_uring_cmd requires io_uring as io_engine
    if args.use_uring_cmd:
        args.use_uring = True

    # write_bench defaults to True (write benchmark), set to False for read benchmark
    write_bench = args.write_bench.lower() in ("true", "1", "yes", "y", "")

    results = []
    # Run LocalDiskBackend benchmark
    if args.backend in ("local_disk", "both"):
        localdisk_bench = LocalDiskBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            local_disk_dir=args.local_disk_dir,
            max_disk_gb=args.max_local_disk_gb,
            use_odirect=args.local_disk_odirect,
            alignment=args.alignment,
            write_bench=write_bench,
            chunk_size=args.chunk_size,
            verify_integrity=args.verify_integrity,
        )
        result = localdisk_bench.run()
        result["local_disk_dir"] = args.local_disk_dir
        results.append(result)

    # Run RustRawBlockBackend benchmark
    if args.backend in ("rust_raw_block", "both"):
        raw_device = args.raw_device
        cleanup_raw_device = False
        if not raw_device:
            # Use same filesystem as local disk for fair comparison
            raw_device = os.path.join(args.local_disk_dir, "raw_block.bin")
            cleanup_raw_device = True

        rustraw_bench = RustRawBlockBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            raw_device=raw_device,
            raw_device_size_gb=args.raw_device_size_gb,
            use_odirect=args.raw_odirect,
            alignment=args.alignment,
            cleanup_raw_device=cleanup_raw_device,
            write_bench=write_bench,
            use_uring=args.use_uring,
            use_uring_cmd=args.use_uring_cmd,
            chunk_size=args.chunk_size,
            verify_integrity=args.verify_integrity,
            max_data_transfer_size=args.max_data_transfer_size,
        )
        result = rustraw_bench.run()
        result["raw_device"] = raw_device
        results.append(result)

    # Run Hf3fsBackend benchmark
    if args.backend in ("hf3fs_backend",):
        hf3fs_bench = Hf3fsBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            remote_url=args.remote_url,
            use_odirect=False,
            alignment=args.alignment,
            write_bench=write_bench,
            chunk_size=args.chunk_size,
            verify_integrity=args.verify_integrity,
        )
        result = hf3fs_bench.run()
        result["hf3fs_dir"] = args.remote_url
        results.append(result)

    # Run FsBackend benchmark
    if args.backend in ("fs_backend",):
        fs_bench = FsBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            remote_url=args.remote_url,
            use_odirect=False,
            alignment=args.alignment,
            write_bench=write_bench,
            chunk_size=args.chunk_size,
            verify_integrity=args.verify_integrity,
        )
        result = fs_bench.run()
        result["fs_dir"] = args.remote_url
        results.append(result)

    # Run BigtableBackend benchmark
    if args.backend in ("bigtable",):
        bigtable_bench = BigtableBackendBenchmark(
            num_ops=args.num_ops,
            concurrency=args.concurrency,
            remote_url=args.remote_url,
            use_odirect=False,
            alignment=args.alignment,
            write_bench=write_bench,
            chunk_size=args.chunk_size,
            verify_integrity=args.verify_integrity,
        )
        result = bigtable_bench.run()
        result["bigtable_table"] = args.remote_url
        results.append(result)

    # Print results
    for result in results:
        print(
            f"{result['backend']}: ops={result['num_ops']} "
            f"concurrency={result['concurrency']} "
            f"write_elapsed={result['write_elapsed_sec']:.3f}s "
            f"write_ops/sec={result['write_ops_per_sec']:.2f}"
        )
        if not write_bench:
            print(
                f"read_elapsed={result['read_elapsed_sec']:.3f}s "
                f"read_ops/sec={result['read_ops_per_sec']:.2f} "
                f"total_elapsed={result['total_elapsed_sec']:.3f}s"
            )
            if args.verify_integrity:
                status = "PASSED" if result["integrity_passed"] else "FAILED"
                print(
                    f"  Integrity check: {status} (errors={result['integrity_errors']})"
                )

    # Write JSON output
    if args.output_json:
        output_path = args.output_json
        if output_path.endswith(os.sep) or os.path.isdir(output_path):
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"storage_backend_io_{ts}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
