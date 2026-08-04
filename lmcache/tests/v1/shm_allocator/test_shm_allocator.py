# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the ShmFile connector.

This test verifies that:
1. Data written via put() is persisted to a file.
2. Data read via get() matches the original data exactly.
3. The C++ subprocess (shm_file_worker) correctly attaches to
   POSIX shared memory and performs I/O without extra copies.
"""

# Standard
from multiprocessing import shared_memory
from typing import Optional
import asyncio
import ctypes
import os
import subprocess
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import LocalCPUBackend

# Local
from ..utils import (
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
)
from .shmfile_connector import ShmFileConnector

logger = init_logger(__name__)

SHM_NAME = "/lmcache_test_shm"
SHM_BUF_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB


_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_BUILD_DIR = os.path.join(_CSRC_DIR, "build")
_WORKER_BIN = os.path.join(_BUILD_DIR, "shm_file_worker")


def _build_worker_binary() -> str:
    """Build shm_file_worker from tests/v1/shm_allocator/csrc/
    and return the path to the compiled binary."""
    env_bin = os.environ.get("SHM_FILE_WORKER_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin

    if os.path.isfile(_WORKER_BIN):
        return _WORKER_BIN

    os.makedirs(_BUILD_DIR, exist_ok=True)
    try:
        subprocess.check_call(
            ["cmake", ".."],
            cwd=_BUILD_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["make", "shm_file_worker"],
            cwd=_BUILD_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip(
            "Cannot compile shm_file_worker. "
            "Ensure cmake and a C++ compiler are available."
        )
    if not os.path.isfile(_WORKER_BIN):
        pytest.skip("shm_file_worker build failed.")
    return _WORKER_BIN


def _get_metadata():
    kv_shape = (32, 2, 256, 8, 128)
    dtype = torch.bfloat16
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=kv_shape,
    )


def _create_shmfs_config(
    storage_dir: str,
    worker_binary: str,
    shm_name: str = SHM_NAME,
    worker_addr: Optional[str] = None,
) -> LMCacheEngineConfig:
    """Build config with shmfs.* extra_config entries."""
    extra = {
        "shmfs.storage_dir": storage_dir,
        "shmfs.shm_name": shm_name,
        "shmfs.worker_binary": worker_binary,
        "shm_name": shm_name,
    }
    if worker_addr:
        extra["shmfs.worker_addr"] = worker_addr
    return LMCacheEngineConfig.from_defaults(
        extra_config=extra,
    )


def _create_local_cpu_backend(memory_allocator):
    config = LMCacheEngineConfig.from_defaults()
    metadata = _get_metadata()
    return LocalCPUBackend(
        config=config,
        metadata=metadata,
        memory_allocator=memory_allocator,
    )


@pytest.fixture
def shm_allocator():
    """Create a MixedMemoryAllocator backed by POSIX shm."""
    allocator = MixedMemoryAllocator(SHM_BUF_SIZE, shm_name=SHM_NAME)
    yield allocator
    allocator.close()


@pytest.fixture
def worker_binary():
    worker_binary_path = _build_worker_binary()
    logger.info(f"Using shm_file_worker binary: {worker_binary_path}")
    return worker_binary_path


class TestShmFileConnector:
    """End-to-end tests for ShmFileConnector."""

    def test_put_and_get_roundtrip(self, shm_allocator, worker_binary):
        """Write data via put(), then read via get() and
        verify byte-for-byte equality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            async_loop, async_thread = init_asyncio_loop()
            try:
                backend = _create_local_cpu_backend(shm_allocator)
                cfg = _create_shmfs_config(
                    tmp_dir,
                    worker_binary,
                )
                connector = ShmFileConnector(
                    loop=async_loop,
                    local_cpu_backend=backend,
                    config=cfg,
                )

                key = dumb_cache_engine_key()

                # Verify key does not exist initially
                fut = asyncio.run_coroutine_threadsafe(
                    connector.exists(key), async_loop
                )
                assert not fut.result()

                # Allocate and fill with deterministic data
                shape = torch.Size([2, 32, 256, 1024])
                dtype = torch.bfloat16
                mem_obj = backend.allocate(shape, dtype)
                assert mem_obj is not None
                mem_obj.ref_count_up()

                torch.manual_seed(42)
                test_data = torch.randint(
                    0,
                    100,
                    mem_obj.raw_data.shape,
                    dtype=torch.int64,
                )
                mem_obj.raw_data.copy_(test_data.to(torch.float32).to(dtype))

                # Put
                fut = asyncio.run_coroutine_threadsafe(
                    connector.put(key, mem_obj), async_loop
                )
                fut.result()

                # Verify file exists on disk
                fut = asyncio.run_coroutine_threadsafe(
                    connector.exists(key), async_loop
                )
                assert fut.result()

                # Get
                fut = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
                retrieved = fut.result()
                assert retrieved is not None

                # Compare
                orig = mem_obj.tensor
                got = retrieved.tensor
                assert orig is not None
                assert got is not None
                assert orig.shape == got.shape
                assert torch.equal(orig, got), "Data mismatch after roundtrip!"

                # Clean up
                fut = asyncio.run_coroutine_threadsafe(connector.close(), async_loop)
                fut.result()
                backend.close()
            finally:
                close_asyncio_loop(async_loop, async_thread)

    def test_get_nonexistent_key(self, shm_allocator, worker_binary):
        """get() on a missing key returns None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            async_loop, async_thread = init_asyncio_loop()
            try:
                backend = _create_local_cpu_backend(shm_allocator)
                cfg = _create_shmfs_config(
                    tmp_dir,
                    worker_binary,
                )
                connector = ShmFileConnector(
                    loop=async_loop,
                    local_cpu_backend=backend,
                    config=cfg,
                )

                key = dumb_cache_engine_key(id=999)
                fut = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
                assert fut.result() is None

                fut = asyncio.run_coroutine_threadsafe(connector.close(), async_loop)
                fut.result()
                backend.close()
            finally:
                close_asyncio_loop(async_loop, async_thread)

    def test_list_keys(self, shm_allocator, worker_binary):
        """list() returns keys for all stored files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            async_loop, async_thread = init_asyncio_loop()
            try:
                backend = _create_local_cpu_backend(shm_allocator)
                cfg = _create_shmfs_config(
                    tmp_dir,
                    worker_binary,
                )
                connector = ShmFileConnector(
                    loop=async_loop,
                    local_cpu_backend=backend,
                    config=cfg,
                )

                shape = torch.Size([2, 32, 256, 1024])
                dtype = torch.bfloat16

                keys = [dumb_cache_engine_key(id=i) for i in range(3)]

                for k in keys:
                    mem_obj = backend.allocate(shape, dtype)
                    assert mem_obj is not None
                    mem_obj.ref_count_up()
                    fut = asyncio.run_coroutine_threadsafe(
                        connector.put(k, mem_obj), async_loop
                    )
                    fut.result()

                fut = asyncio.run_coroutine_threadsafe(connector.list(), async_loop)
                listed = set(fut.result())
                expected = {k.to_string() for k in keys}
                assert listed == expected

                fut = asyncio.run_coroutine_threadsafe(connector.close(), async_loop)
                fut.result()
                backend.close()
            finally:
                close_asyncio_loop(async_loop, async_thread)


class TestShmWorkerSubprocess:
    """Test the C++ shm_file_worker directly."""

    def test_allocator_receives_shm_name_from_config(self):
        """Verify that initialize_allocator passes shm_name
        when extra_config contains shm_name."""
        config = LMCacheEngineConfig.from_defaults(
            extra_config={"shm_name": "/lmc_prod_test"},
            max_local_cpu_size=0.001,
        )
        metadata = _get_metadata()
        backend = LocalCPUBackend(config=config, metadata=metadata)
        alloc = backend.memory_allocator
        assert isinstance(alloc, MixedMemoryAllocator)
        assert alloc.shm_name == "/lmc_prod_test"
        backend.close()

    def test_allocator_no_shm_name_without_extra_config(self):
        """Verify that shm_name is None when extra_config
        does not contain shm_name."""
        config = LMCacheEngineConfig.from_defaults(
            max_local_cpu_size=0.001,
        )
        metadata = _get_metadata()
        backend = LocalCPUBackend(config=config, metadata=metadata)
        alloc = backend.memory_allocator
        assert isinstance(alloc, MixedMemoryAllocator)
        assert alloc.shm_name is None
        backend.close()

    def test_worker_attach_and_io(self, worker_binary):
        """Verify the worker can attach, write, and read."""
        shm_test_name = "lmcache_worker_test"
        shm_test_size = 4096

        # Create shm segment from Python side
        shm = shared_memory.SharedMemory(
            name=shm_test_name, create=True, size=shm_test_size
        )

        try:
            # Write known pattern into shm
            msg = b"Hello SHM Worker!"
            shm.buf[:17] = msg

            # Get the base address of shm buffer
            arr_t = ctypes.c_uint8 * shm_test_size
            arr = arr_t.from_buffer(shm.buf)
            base_addr = ctypes.addressof(arr)

            proc = subprocess.Popen(
                [worker_binary],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            def send(cmd):
                assert proc.stdin is not None
                assert proc.stdout is not None
                proc.stdin.write(cmd + "\n")
                proc.stdin.flush()
                return proc.stdout.readline().strip()

            # Attach with base_addr
            resp = send("ATTACH /%s %d %d" % (shm_test_name, shm_test_size, base_addr))
            assert resp == "OK", "ATTACH failed: %s" % resp

            # Write shm content to a temp file (data_ptr = base_addr)
            with tempfile.NamedTemporaryFile(delete=False) as f:
                tmp_path = f.name
            try:
                resp = send("WRITE %s %d 17" % (tmp_path, base_addr))
                assert resp.startswith("OK"), "WRITE failed: %s" % resp

                with open(tmp_path, "rb") as f:
                    assert f.read() == msg

                # Now zero the shm region and read back
                shm.buf[:17] = b"\x00" * 17
                resp = send("READ %s %d 17" % (tmp_path, base_addr))
                assert resp.startswith("OK"), "READ failed: %s" % resp
                assert bytes(shm.buf[:17]) == msg
            finally:
                os.unlink(tmp_path)

            send("QUIT")
            proc.wait(timeout=5)
        finally:
            del arr  # Release exported pointer before close
            shm.close()
            shm.unlink()

    def test_worker_tcp_mode(self, worker_binary):
        """Verify the worker can run in TCP mode and
        communicate via socket."""
        shm_test_name = "lmcache_tcp_test"
        shm_test_size = 4096
        tcp_port = 19800

        shm = shared_memory.SharedMemory(
            name=shm_test_name,
            create=True,
            size=shm_test_size,
        )

        try:
            msg = b"Hello TCP Worker!"
            shm.buf[:17] = msg

            arr_t = ctypes.c_uint8 * shm_test_size
            arr = arr_t.from_buffer(shm.buf)
            base_addr = ctypes.addressof(arr)

            # Start worker in TCP mode
            proc = subprocess.Popen(
                [
                    worker_binary,
                    "--listen",
                    "127.0.0.1:%d" % tcp_port,
                ],
                stderr=subprocess.PIPE,
            )

            # Wait for worker to start listening
            time.sleep(0.5)

            # Standard
            import socket as sock_mod

            conn = sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM)
            conn.connect(("127.0.0.1", tcp_port))

            def send_tcp(cmd):
                conn.sendall((cmd + "\n").encode("utf-8"))
                buf = b""
                while b"\n" not in buf:
                    data = conn.recv(4096)
                    if not data:
                        return "ERROR connection closed"
                    buf += data
                return buf.split(b"\n", 1)[0].decode().strip()

            # Attach
            resp = send_tcp(
                "ATTACH /%s %d %d" % (shm_test_name, shm_test_size, base_addr)
            )
            assert resp == "OK", "ATTACH failed: %s" % resp

            # Write to file
            with tempfile.NamedTemporaryFile(delete=False) as f:
                tmp_path = f.name
            try:
                resp = send_tcp("WRITE %s %d 17" % (tmp_path, base_addr))
                assert resp.startswith("OK"), "WRITE failed: %s" % resp

                with open(tmp_path, "rb") as f:
                    assert f.read() == msg

                # Zero and read back
                shm.buf[:17] = b"\x00" * 17
                resp = send_tcp("READ %s %d 17" % (tmp_path, base_addr))
                assert resp.startswith("OK"), "READ failed: %s" % resp
                assert bytes(shm.buf[:17]) == msg
            finally:
                os.unlink(tmp_path)

            send_tcp("QUIT")
            conn.close()
            proc.wait(timeout=5)
        finally:
            del arr
            shm.close()
            shm.unlink()
