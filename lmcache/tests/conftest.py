# SPDX-License-Identifier: Apache-2.0
# Standard
# Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
#
# 2026/4/20 Add mock for hf3fs_fuse
#   Wenwen Chen <wenwen.chen@samsung.com>

# Standard
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional
from unittest.mock import patch
import asyncio
import importlib.util
import os
import random
import shlex
import socket
import subprocess
import time

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata

if importlib.util.find_spec("pytest_benchmark") is None:

    @pytest.fixture
    def benchmark():
        pytest.skip("pytest-benchmark is not installed")


@pytest.fixture(scope="session", autouse=True)
def disable_usage_tracking():
    """Keep the test suite from sending usage telemetry to the stats server.

    Tests that exercise the telemetry itself (tests/test_usage_context.py)
    re-enable it per-test via monkeypatch with an injected transport.
    """
    os.environ["LMCACHE_TRACK_USAGE"] = "false"
    yield


# This is to mock the constructor and destructor of
# MixedMemoryAllocator and PinMemoryAllocator to
# use pin_memory=True for their constructors and
# avoid calling cudaHostRegister and cudaHostUnregister
# which may throw an error if torch.empty returns a buffer
# that cannot be registered (which happens quicker on some machines,
# especially when torch is doing many allocations and frees)


# In production, using the cuda C++ API gives us a larger pinned buffer
# but for the tests, we do not need this so this mock leaves the unit tests
# functionally the same
"""
@pytest.fixture(autouse=True, scope="session")
def patch_mixed_allocator():
    def fake_mixed_init(self, size: int, use_paging: bool = False, **kwargs):
        # self.buffer = torch.empty(size, dtype=torch.uint8)
        # ptr = self.buffer.data_ptr()
        # err = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
        # assert err == 0, (
        #     f"cudaHostRegister failed: {torch.cuda.cudart().cudaGetErrorString(err)}"
        # )
        self._unregistered = False
        self.buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        if use_paging:
            assert "shape" in kwargs, (
                "shape must be specified for paged memory allocator"
            )
            assert "dtype" in kwargs, (
                "dtype must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.pin_allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            self.pin_allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

        self.buffer_allocator = BufferAllocator("cpu")

    def fake_mixed_close(self):
        if not self._unregistered:
            torch_dev.synchronize()
            # torch.cuda.cudart().cudaHostUnregister(self.buffer.data_ptr())
            self._unregistered = True

    with (
        patch(
            "lmcache.v1.memory_allocators.mixed_memory_allocator.MixedMemoryAllocator.__init__",
            fake_mixed_init,
        ),
        patch(
            "lmcache.v1.memory_allocators.mixed_memory_allocator.MixedMemoryAllocator.close",
            fake_mixed_close,
        ),
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_pin_allocator():
    def fake_pin_init(self, size: int, use_paging: bool = False, **kwargs):

        # self.buffer = torch.empty(size, dtype=torch.uint8)
        # ptr = self.buffer.data_ptr()
        # err = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
        # assert err == 0, (
        #     f"cudaHostRegister failed: {torch.cuda.cudart().cudaGetErrorString(err)}"
        # )
        self._unregistered = False
        self.buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        if use_paging:
            assert "shape" in kwargs, (
                "shape must be specified for paged memory allocator"
            )
            assert "dtype" in kwargs, (
                "dtype must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    def fake_pin_close(self):
        if not self._unregistered:
            torch_dev.synchronize()
            # torch.cuda.cudart().cudaHostUnregister(self.buffer.data_ptr())
            self._unregistered = True

    with (
        patch(
            "lmcache.v1.memory_allocators.pin_memory_allocator.PinMemoryAllocator.__init__",
            fake_pin_init,
        ),
        patch(
            "lmcache.v1.memory_allocators.pin_memory_allocator.PinMemoryAllocator.close",
            fake_pin_close,
        ),
    ):
        yield
"""


class MockSyncGlideClient:
    """In-memory mock of a synchronous Glide (Valkey) client."""

    _store: dict[bytes, bytes] = {}
    #: Records the ``expiry`` argument from the most recent ``set`` call so
    #: tests can assert TTL behavior. ``None`` means no expiry was passed.
    last_expiry = None

    def set(self, key: bytes, value, expiry=None) -> None:
        self._store[key] = bytes(value)
        type(self).last_expiry = expiry

    def get(self, key: bytes, buffer=None):
        data = self._store.get(key)
        if buffer is None:
            return data
        # Mirror glide's native buffer-GET: write into the caller's buffer and
        # return the number of bytes read (an int count), or None on miss.
        if data is None:
            return None
        n = len(data)
        buffer[:n] = data
        return n

    def exists(self, keys: list[bytes]) -> int:
        return sum(1 for k in keys if k in self._store)

    def delete(self, keys: list[bytes]) -> int:
        removed = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                removed += 1
        return removed

    @classmethod
    def reset_store(cls) -> None:
        cls._store.clear()
        cls.last_expiry = None


class MockRedis:
    def __init__(
        self, host=None, port=None, url=None, decode_responses=False, **kwargs
    ):
        self.store = {}
        self.host = host
        self.port = port
        self.url = url
        self.decode_responses = decode_responses

    def set(self, key, value):
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key, None)

    def exists(self, key):
        return key in self.store

    def delete(self, key):
        return self.store.pop(key, None) is not None

    def scan(self, cursor=0, match=None):
        keys = [s.encode("utf-8") for s in self.store.keys()]
        return (0, keys)

    def close(self):
        pass

    @classmethod
    def from_url(cls, url, decode_responses=False, **kwargs):
        """Mock implementation of Redis.from_url"""
        return cls(url=url, decode_responses=decode_responses, **kwargs)

    @classmethod
    def from_pool(cls, pool, **kwargs):
        """Mock implementation of Redis.from_pool"""
        return cls(**kwargs)


class MockAsyncRedis(MockRedis):
    """Async version of MockRedis"""

    async def set(self, key, value):
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key, None)

    async def exists(self, key):
        return key in self.store

    async def delete(self, key):
        return self.store.pop(key, None) is not None

    async def close(self):
        pass

    @classmethod
    def from_url(cls, url, decode_responses=False, **kwargs):
        """Mock implementation of Redis.from_url"""
        return cls(url=url, decode_responses=decode_responses, **kwargs)

    @classmethod
    def from_pool(cls, pool, **kwargs):
        """Mock implementation of Redis.from_pool"""
        return cls(**kwargs)


class MockRedisSentinel:
    def __init__(self, hosts_and_ports, socket_timeout=None, **kwargs):
        self.hosts_and_ports = hosts_and_ports
        self.socket_timeout = socket_timeout
        # Create a shared store but separate instances for master/slave
        self.shared_store = {}
        self.master_redis = MockRedis()
        self.slave_redis = MockRedis()
        # Share the store between master and slave to simulate Redis Sentinel behavior
        self.master_redis.store = self.shared_store
        self.slave_redis.store = self.shared_store

    def master_for(
        self, service_name, socket_timeout=None, username=None, password=None, **kwargs
    ):
        return self.master_redis

    def slave_for(
        self, service_name, socket_timeout=None, username=None, password=None, **kwargs
    ):
        return self.slave_redis


class MockRESPClient:
    """In-memory mock of RESPClient so RESP connector tests never hit real Redis."""

    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int,
        loop=None,
        username: str = "",
        password: str = "",
    ):
        self._store: dict[str, bytes] = {}  # key -> bytes
        self._loop = loop
        self._closed = False
        self._username = username
        self._password = password

    async def exists(self, key: str) -> bool:
        return key in self._store

    def exists_sync(self, key: str) -> bool:
        if self._loop is None:
            return key in self._store
        fut = asyncio.run_coroutine_threadsafe(self.exists(key), self._loop)
        return fut.result(timeout=10.0)

    def _copy_into_buf(self, buf: memoryview, data: bytes) -> None:
        """
        Copy bytes into buffer; buf may be non-byte or multi-dimensional
        (e.g. from bfloat16 tensor).
        """
        view = buf.cast("B")
        n = len(data)
        try:
            view[:n] = data
        except (NotImplementedError, TypeError, ValueError):
            # Multi-dimensional or non-contiguous:
            # write via flat numpy view (same memory)
            arr = np.asarray(view, dtype=np.uint8, copy=False)
            arr.flat[:n] = np.frombuffer(data, dtype=np.uint8, count=n)

    async def get(self, key: str, buf: memoryview) -> None:
        data = self._store.get(key)
        if data is None:
            raise RuntimeError("key not found")
        self._copy_into_buf(buf, data)

    async def set(self, key: str, buf: memoryview) -> None:
        self._store[key] = bytes(buf.cast("B"))

    async def batch_get(self, keys: list, bufs: list) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        for k, b in zip(keys, bufs, strict=False):
            data = self._store.get(k)
            if data is None:
                raise RuntimeError("key not found")
            self._copy_into_buf(b, data)

    async def batch_set(self, keys: list, bufs: list) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        for k, b in zip(keys, bufs, strict=False):
            self._store[k] = bytes(b.cast("B"))

    async def batch_exists(self, keys: list) -> list:
        return [k in self._store for k in keys]

    def batch_exists_sync(self, keys: list) -> list:
        if self._loop is None:
            return [k in self._store for k in keys]
        fut = asyncio.run_coroutine_threadsafe(self.batch_exists(keys), self._loop)
        return fut.result(timeout=10.0)

    def close(self) -> None:
        self._closed = True
        self._store.clear()


class MockRedisCluster:
    def __init__(
        self, startup_nodes=None, max_connections=None, decode_responses=False, **kwargs
    ):
        self.startup_nodes = startup_nodes or []
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        self.store = {}

    async def set(self, key, value):
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key, None)

    async def exists(self, key):
        return key in self.store

    async def delete(self, key):
        return self.store.pop(key, None) is not None

    async def close(self):
        pass


class MockHf3fsIoring:
    """Mock I/O ring that stores data to local temp storage.

    This mock implements the same interface as the real hf3fs ioring,
    but uses local filesystem for actual I/O operations.
    """

    def __init__(
        self,
        mount_point: str,
        entries: int,
        for_read: bool = True,
        timeout: int = 200,
        numa: int = -1,
    ):
        self.mount_point = mount_point
        self.entries = entries
        self.for_read = for_read
        self.timeout = timeout
        self.numa = numa
        self._prepared_iov = None
        self._prepared_is_read: Optional[bool] = None
        self._prepared_fd: Optional[int] = None
        self._prepared_offset: Optional[int] = None
        self._result_bytes = 0

    def prepare(self, iov, is_read: bool, fd: int, offset: int):
        """Prepare I/O operation - stores parameters for later use."""
        self._prepared_iov = iov
        self._prepared_is_read = is_read
        self._prepared_fd = fd
        self._prepared_offset = offset

    def submit(self):
        """Submit I/O operation - performs actual file I/O.

        For write operations: writes data from shared memory to file.
        For read operations: reads data from file into shared memory.
        """
        # Standard
        import sys

        if self._prepared_iov is None or self._prepared_fd is None:
            return self

        try:
            # Get the shared memory buffer from iov
            shm = self._prepared_iov.shm
            if shm is None:
                print(
                    f"MockHf3fsIoring.submit: shm is None, "
                    f"_prepared_iov={self._prepared_iov}",
                    file=sys.stderr,
                )
                return self

            # Get the length of data to transfer
            slice_start = 0
            if hasattr(self._prepared_iov, "slice_start"):
                slice_start = self._prepared_iov.slice_start
            slice_end = None
            if hasattr(self._prepared_iov, "slice_end"):
                slice_end = self._prepared_iov.slice_end

            if slice_end is not None:
                length = slice_end - slice_start
            elif slice_start > 0:
                length = shm.size - slice_start if hasattr(shm, "size") else 4096
            else:
                length = shm.size if hasattr(shm, "size") else 4096

            if self._prepared_is_read:
                os.lseek(self._prepared_fd, self._prepared_offset, os.SEEK_SET)
                # Read data from file
                data = os.read(self._prepared_fd, length)
                # Write data to shared memory using slice range
                if slice_end:
                    shm.buf[slice_start:slice_end] = data
                else:
                    shm.buf[slice_start:] = data
                self._result_bytes = len(data)
            else:
                # Get data from shared memory using slice range
                if slice_end:
                    data = bytes(shm.buf[slice_start:slice_end])
                else:
                    data = bytes(shm.buf[slice_start:])
                os.lseek(self._prepared_fd, self._prepared_offset, os.SEEK_SET)
                # Write data to file
                bytes_written = os.write(self._prepared_fd, data)
                self._result_bytes = bytes_written

        except Exception as e:
            # Log error but don't fail - return 0 bytes
            # Standard
            import traceback

            print(f"MockHf3fsIoring.submit error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._result_bytes = 0

        return self

    def wait(self, min_results: int = 1):
        """Wait for I/O completion and return results.

        Returns:
            List of MockIorResult, each with .result containing bytes transferred
        """

        class MockIorResult:
            def __init__(self, result: int):
                self.result = result

        return [MockIorResult(self._result_bytes)]


class MockHf3fsIovec:
    """Mock I/O vector for HF3FS operations."""

    def __init__(
        self,
        shm: SharedMemory,
        hf3fs_mount_point: str,
        numa: int = -1,
        slice_start: int = 0,
        slice_end: Optional[int] = None,
    ):
        self.shm = shm
        self.mount_point = hf3fs_mount_point
        self.numa = numa
        self.slice_start = slice_start
        self.slice_end = slice_end  # None means full range

    def __getitem__(self, index):
        # Support slicing operation, e.g., iov[:length]
        if isinstance(index, slice):
            start = index.start or 0
            end = index.stop
            return MockHf3fsIovec(self.shm, self.mount_point, self.numa, start, end)
        return self

    def __len__(self):
        if self.slice_end is not None:
            return self.slice_end - self.slice_start
        elif self.slice_start > 0:
            return getattr(self.shm, "size", 4096) - self.slice_start
        return getattr(self.shm, "size", 4096)


class MockHf3fsFuse:
    """Mock for hf3fs_fuse.io module functions."""

    @staticmethod
    def extract_mount_point(path: str) -> str:
        """Mock extract_mount_point - returns the path as-is if it exists.

        In real 3FS, this function extracts the actual 3FS mount point from a path.
        For testing, we just return the path if it exists, or return as-is.
        """
        p = Path(path)
        if p.exists():
            return str(p.resolve())
        return path

    @staticmethod
    def make_iovec(shm: SharedMemory, hf3fs_mount_point: str, numa: int = -1):
        """Mock make_iovec - creates a mock I/O vector."""
        return MockHf3fsIovec(shm, hf3fs_mount_point, numa)

    @staticmethod
    def make_ioring(
        mount_point: str,
        entries: int,
        for_read: bool = True,
        timeout: int = 200,
        numa: int = -1,
    ):
        """Mock make_ioring - creates a mock I/O ring."""
        return MockHf3fsIoring(mount_point, entries, for_read, timeout, numa)

    @staticmethod
    def register_fd(fd: int):
        """Mock register_fd - no-op for testing."""
        pass

    @staticmethod
    def deregister_fd(fd: int):
        """Mock deregister_fd - no-op for testing."""
        pass


@dataclass
class LMCacheServerProcess:
    server_url: str
    server_process: object


@pytest.fixture(scope="function", autouse=True)
def mock_redis():
    with (
        patch("redis.Redis", MockRedis) as mock_redis_class,
        patch("redis.from_url", MockRedis.from_url),
        patch("redis.asyncio.Redis", MockAsyncRedis),
        patch("redis.asyncio.from_url", MockAsyncRedis.from_url),
        patch("redis.asyncio.ConnectionPool.from_url", lambda url, **kwargs: None),
        patch("redis.asyncio.Redis.from_pool", MockAsyncRedis.from_pool),
    ):
        yield mock_redis_class


@pytest.fixture(scope="function", autouse=True)
def mock_redis_sentinel():
    with (
        patch("redis.Sentinel", MockRedisSentinel) as mock,
        patch("redis.asyncio.Sentinel", MockRedisSentinel),
    ):
        yield mock


@pytest.fixture(scope="function", autouse=True)
def mock_redis_cluster():
    with patch("redis.asyncio.cluster.RedisCluster", MockRedisCluster) as mock:
        yield mock


@pytest.fixture(scope="function", autouse=True)
def mock_hf3fs():
    """Mock hf3fs_fuse.io module for testing without 3FS filesystem.

    This fixture creates a fake hf3fs_fuse.io module in sys.modules before
    any imports happen, allowing tests to run without the actual package.
    """
    # Standard
    from types import ModuleType
    import sys

    # Create a fake hf3fs_fuse module if it doesn't exist
    hf3fs_fuse = ModuleType("hf3fs_fuse")
    hf3fs_fuse.__file__ = "/mock/hf3fs_fuse/__init__.py"

    # Create hf3fs_fuse.io module
    hf3fs_io = ModuleType("hf3fs_fuse.io")
    hf3fs_io.__file__ = "/mock/hf3fs_fuse/io.py"

    # Add mock functions to hf3fs_fuse.io
    hf3fs_io.extract_mount_point = MockHf3fsFuse.extract_mount_point
    hf3fs_io.make_iovec = MockHf3fsFuse.make_iovec
    hf3fs_io.make_ioring = MockHf3fsFuse.make_ioring
    hf3fs_io.register_fd = MockHf3fsFuse.register_fd
    hf3fs_io.deregister_fd = MockHf3fsFuse.deregister_fd

    # Register modules in sys.modules
    sys.modules["hf3fs_fuse"] = hf3fs_fuse
    sys.modules["hf3fs_fuse.io"] = hf3fs_io

    yield

    # Cleanup: remove the mock modules after test
    sys.modules.pop("hf3fs_fuse", None)
    sys.modules.pop("hf3fs_fuse.io", None)


@pytest.fixture(scope="module")
def lmserver_v1_process(request):
    def ensure_connection(host, port):
        retries = 10
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        successful = False
        while retries > 0:
            retries -= 1
            try:
                print("Probing connection, remaining retries: ", retries)
                client_socket.connect((host, port))
                successful = True
                break
            except ConnectionRefusedError:
                time.sleep(1)
                print("Connection refused!")
                continue
            except Exception as e:
                print(f"other Exception: {e}")
                continue

        client_socket.close()
        return successful

    # Specify remote device
    device = request.param

    # Start the process
    max_retries = 5
    while max_retries > 0:
        max_retries -= 1
        port_number = random.randint(10000, 65500)
        print("Starting the lmcache v1 server process on port")
        proc = subprocess.Popen(
            shlex.split(
                f"python3 -m lmcache.v1.server localhost {port_number} {device}"
            )
        )

        # Wait for lmcache process to start
        time.sleep(5)

        successful = False
        if proc.poll() is not None:
            successful = True
        else:
            successful = ensure_connection("localhost", port_number)

        if not successful:
            proc.terminate()
            proc.wait()
        else:
            break

    # Yield control back to the test until it finishes
    server_url = f"lm://localhost:{port_number}"
    yield LMCacheServerProcess(server_url, proc)

    # Terminate the process
    proc.terminate()
    proc.wait()

    # Destroy remote disk path
    if device not in ["cpu"]:
        subprocess.run(shlex.split(f"rm -rf {device}"))


@pytest.fixture(scope="function")
def autorelease(request):
    objects = []

    def _factory(obj):
        objects.append(obj)
        return obj

    yield _factory

    # Cleanup all objects created by the factory
    for obj in objects:
        obj.close()


@pytest.fixture(scope="function")
def autorelease_v1(request):
    objects = []

    def _factory(obj, **kwargs):
        if isinstance(obj, LMCacheEngine):
            obj.post_init(**kwargs)
        objects.append(obj)
        return obj

    yield _factory

    LMCacheEngineBuilder.destroy("test")

    # Cleanup all objects created by the factory
    # IMPORTANT: We must close connectors to ensure AsyncPQExecutor and other
    # async resources are properly cleaned up
    # NOTE: Skip LMCacheEngine instances since destroy() already calls close()
    for obj in objects:
        if isinstance(obj, LMCacheEngine):
            continue
        try:
            # Check if object has a close method
            if hasattr(obj, "close"):
                obj.close()
        except Exception as e:
            # Log but don't fail the test
            print("Error during close obj:%s - %s", obj, e)


@pytest.fixture(scope="session")
def memory_allocator():
    """One MixedMemoryAllocator (5GB) for the whole test session;
    .close() is a no-op per-test."""
    _real = MixedMemoryAllocator(5 * 1024 * 1024 * 1024)  # 5GB

    class _NoCloseWrapper:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            # No-op so per-test close() calls don't shut down the shared allocator
            pass

    try:
        yield _NoCloseWrapper(_real)
    finally:
        # Actually close once when the session ends
        _real.close()


@pytest.fixture(autouse=True)  # function-scoped by default
def use_shared_allocator(request, monkeypatch, memory_allocator):
    """Default: patch. Opt out with @pytest.mark.no_shared_allocator."""
    if request.node.get_closest_marker("no_shared_allocator"):
        # do NOT patch for this test
        yield
        return

    def _create_shared_allocator(config, metadata, numa_mapping):
        return memory_allocator

    monkeypatch.setattr(
        LMCacheEngineBuilder,
        "_Create_memory_allocator",
        _create_shared_allocator,
    )
    yield


@pytest.fixture(scope="function")
def lmcache_engine_metadata(role="worker"):
    """Create a fresh LMCacheMetadata for each test."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(32, 2, 256, 32, 128),
        use_mla=False,
        role=role,
    )


@pytest.fixture(scope="session")
def bigtable_emulator():
    """Start or connect to the Bigtable emulator for integration testing."""
    # Standard
    import os
    import shutil
    import subprocess
    import time

    existing_host = os.environ.get("BIGTABLE_EMULATOR_HOST")
    if existing_host:
        print(f"\n[Fixture] Reusing existing Bigtable emulator at {existing_host}...")
        yield existing_host
        return

    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping Bigtable emulator integration tests in CI")

    if not shutil.which("gcloud"):
        pytest.skip(
            "gcloud CLI not found, skipping Bigtable Emulator integration tests"
        )

    port = "8899"
    host_port = f"localhost:{port}"

    print(f"\n[Fixture] Starting Bigtable emulator on {host_port}...")
    emulator_process = subprocess.Popen(
        [
            "gcloud",
            "beta",
            "emulators",
            "bigtable",
            "start",
            f"--host-port={host_port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    os.environ["BIGTABLE_EMULATOR_HOST"] = host_port

    def is_port_open(h: str, p: int, t: float = 1.0) -> bool:
        # Standard
        import socket

        try:
            with socket.create_connection((h, p), timeout=t):
                return True
        except OSError:
            return False

    # Wait for the emulator to initialize
    port_num = int(port)
    start_time = time.time()
    success = False
    while time.time() - start_time < 10.0:
        if is_port_open("localhost", port_num):
            success = True
            break
        time.sleep(0.5)

    if not success:
        print(
            f"\n[Fixture] Bigtable emulator failed to bind to {host_port} within 10s!"
        )
        try:
            stdout, stderr = emulator_process.communicate(timeout=2.0)
            print(f"Stdout:\n{stdout.decode('utf-8', errors='ignore')}")
            print(f"Stderr:\n{stderr.decode('utf-8', errors='ignore')}")
        except Exception as e:
            print(f"Could not retrieve process logs: {e}")
        emulator_process.kill()
        pytest.fail("Failed to start Bigtable emulator")

    yield host_port

    print("\n[Fixture] Stopping Bigtable emulator...")
    emulator_process.terminate()
    try:
        emulator_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        emulator_process.kill()

    if "BIGTABLE_EMULATOR_HOST" in os.environ:
        del os.environ["BIGTABLE_EMULATOR_HOST"]
