# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import multiprocessing as mp

import numpy as np
import pytest
import torch

from vllm.renderers.paged_shm.client import PagedShmClient
from vllm.renderers.paged_shm.server import zmq_server
from vllm.utils import random_uuid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def server_address():
    """
    Spawn a real PagedShmServer in a subprocess and return its IPC address.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    stop_event = ctx.Event()

    proc = ctx.Process(
        target=zmq_server,
        args=(1024 * 1024, 4096, child_conn, stop_event),
    )
    proc.start()
    address = parent_conn.recv()
    parent_conn.close()

    yield address

    stop_event.set()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join()


@pytest.fixture(scope="function")
def client(server_address):
    """
    Create a fresh PagedShmClient connected to the test server.
    """
    c = PagedShmClient(address=server_address, pin=False)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_uuid() -> str:
    """Return a short unique identifier for test items."""
    return f"test-{random_uuid()}"


def _blocks_needed(size: int, block_size: int = 4096) -> int:
    """Return the number of blocks required to store `size` bytes."""
    return math.ceil(size / block_size)


# ---------------------------------------------------------------------------
# Basic write / read
# ---------------------------------------------------------------------------


class TestWriteRead:
    """Verify round‑trip correctness for various data types."""

    def test_write_read_bytes(self, client):
        uuid = _unique_uuid()
        data = b"Hello, shared memory!"
        state_before = client.get_manager_state()

        client.write(uuid, data)

        state_after_write = client.get_manager_state()
        needed = _blocks_needed(len(data))
        assert (
            state_after_write["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )
        assert (
            state_after_write["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = client.read(uuid)
        assert isinstance(result, np.ndarray)
        assert result.tobytes() == data

        client.delete(uuid)
        state_final = client.get_manager_state()
        assert state_final["cached_items_count"] == state_before["cached_items_count"]
        assert state_final["free_blocks_count"] == state_before["free_blocks_count"]

    def test_write_read_numpy(self, client):
        uuid = _unique_uuid()
        original = np.arange(100, dtype=np.float32)
        state_before = client.get_manager_state()

        client.write(uuid, original)

        state_after_write = client.get_manager_state()
        needed = _blocks_needed(original.nbytes)
        assert (
            state_after_write["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = client.read(uuid)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result.view(np.float32), original)

        client.delete(uuid)
        state_final = client.get_manager_state()
        assert state_final["cached_items_count"] == state_before["cached_items_count"]

    def test_write_read_torch_cpu(self, client):
        uuid = _unique_uuid()
        original = torch.arange(50, dtype=torch.int32)
        state_before = client.get_manager_state()

        client.write(uuid, original)

        state_after_write = client.get_manager_state()
        needed = _blocks_needed(original.numel() * original.element_size())
        assert (
            state_after_write["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result_np = client.read(uuid)
        assert isinstance(result_np, np.ndarray)
        result = torch.from_numpy(result_np)
        torch.testing.assert_close(result.view(torch.int32)[: len(original)], original)

        client.delete(uuid)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_write_read_torch_gpu(self, server_address):
        client = PagedShmClient(address=server_address, pin=True)
        try:
            uuid = _unique_uuid()
            original = torch.randint(0, 255, (500,), dtype=torch.uint8, device="cuda")
            state_before = client.get_manager_state()

            client.write(uuid, original)

            state_after_write = client.get_manager_state()
            needed = _blocks_needed(original.numel())
            assert (
                state_after_write["cached_items_count"]
                == state_before["cached_items_count"] + 1
            )
            assert (
                state_after_write["free_blocks_count"]
                == state_before["free_blocks_count"] - needed
            )

            result = client.read(uuid, device="cuda")
            assert isinstance(result, torch.Tensor)
            torch.testing.assert_close(result, original)

            client.delete(uuid)
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Large data (spanning multiple blocks)
# ---------------------------------------------------------------------------


class TestMultiBlock:
    """Test data that spans several shared memory blocks (block size = 4096)."""

    @pytest.mark.parametrize("size", [8000, 16384, 20000])
    def test_bytes_multi_block(self, client, size):
        uuid = _unique_uuid()
        data = bytes(np.random.bytes(size))
        state_before = client.get_manager_state()

        client.write(uuid, data)

        state_after_write = client.get_manager_state()
        needed = _blocks_needed(size)
        assert (
            state_after_write["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = client.read(uuid)
        assert result.tobytes() == data

        client.delete(uuid)
        state_final = client.get_manager_state()
        assert state_final["free_blocks_count"] == state_before["free_blocks_count"]

    @pytest.mark.parametrize("size", [8000, 16384, 20000])
    def test_numpy_multi_block(self, client, size):
        uuid = _unique_uuid()
        original = np.random.randint(0, 256, size, dtype=np.uint8)
        state_before = client.get_manager_state()

        client.write(uuid, original)

        state_after_write = client.get_manager_state()
        needed = _blocks_needed(original.nbytes)
        assert (
            state_after_write["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = client.read(uuid)
        np.testing.assert_array_equal(result, original)

        client.delete(uuid)

    @pytest.mark.parametrize("size", [8000, 16384, 20000])
    def test_torch_multi_block(self, client, size):
        uuid = _unique_uuid()
        original = torch.randint(0, 256, (size,), dtype=torch.uint8)
        state_before = client.get_manager_state()

        client.write(uuid, original)

        state_after_write = client.get_manager_state()
        needed = _blocks_needed(size)
        assert (
            state_after_write["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result_np = client.read(uuid)
        result = torch.from_numpy(result_np)
        torch.testing.assert_close(result, original)

        client.delete(uuid)


# ---------------------------------------------------------------------------
# Context manager usage
# ---------------------------------------------------------------------------


class TestContextManagers:
    """Exercise the write/read context managers directly."""

    def test_write_context_commit(self, client):
        uuid = _unique_uuid()
        data = b"context write test"
        size = len(data)
        state_before = client.get_manager_state()

        with client.write_context(uuid, size) as ctx:
            client._storage.write(data, ctx.blocks)

        state_after_commit = client.get_manager_state()
        needed = _blocks_needed(size)
        assert (
            state_after_commit["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )
        assert (
            state_after_commit["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = client.read(uuid)
        assert result.tobytes() == data

        client.delete(uuid)

    def test_write_context_rollback(self, client):
        uuid = _unique_uuid()
        data = b"should not be visible"
        size = len(data)
        state_before = client.get_manager_state()

        class TestException(Exception):
            pass

        with pytest.raises(TestException):
            with client.write_context(uuid, size) as ctx:
                client._storage.write(data, ctx.blocks)
                raise TestException("trigger rollback")

        state_after_rollback = client.get_manager_state()
        assert (
            state_after_rollback["free_blocks_count"]
            == state_before["free_blocks_count"]
        )
        assert (
            state_after_rollback["cached_items_count"]
            == state_before["cached_items_count"]
        )

        with pytest.raises(RuntimeError, match="Server error"):
            client.read(uuid)

    def test_read_context(self, client):
        uuid = _unique_uuid()
        data = b"read context test"
        client.write(uuid, data)

        state_before_read = client.get_manager_state()
        reading_before = state_before_read["reading_items_count"]

        with client.read_context(uuid) as ctx:
            state_during = client.get_manager_state()
            assert state_during["reading_items_count"] == reading_before + 1
            assert ctx.size == len(data)
            result = client._storage.read_to_numpy(ctx.size, ctx.blocks)
            assert result.tobytes() == data

        state_after = client.get_manager_state()
        assert state_after["reading_items_count"] == reading_before

        client.delete(uuid)

    def test_iterator_numpy_context(self, client):
        uuid = _unique_uuid()
        original = np.random.randint(0, 256, 10000, dtype=np.uint8)
        client.write(uuid, original)

        with client.get_iterator_numpy(uuid) as it:
            blocks = []
            for arr, valid_len in it:
                blocks.append(arr[:valid_len])
        assembled = np.concatenate(blocks)
        np.testing.assert_array_equal(assembled, original)

        client.delete(uuid)

    def test_iterator_tensor_context(self, client):
        uuid = _unique_uuid()
        original = torch.randint(0, 256, (10000,), dtype=torch.uint8)
        client.write(uuid, original)

        with client.get_iterator_tensor(uuid) as it:
            blocks = []
            for tensor, valid_len in it:
                blocks.append(tensor[:valid_len])
        assembled = torch.cat(blocks)
        torch.testing.assert_close(assembled, original)

        client.delete(uuid)


# ---------------------------------------------------------------------------
# Error and edge cases
# ---------------------------------------------------------------------------


class TestErrors:
    """Validate appropriate error handling."""

    def test_read_nonexistent_uuid(self, client):
        with pytest.raises(RuntimeError, match="Server error"):
            client.read("nonexistent-uuid")

    def test_write_exceeding_block_count(self, client):
        uuid = _unique_uuid()
        too_large = bytes(1024 * 1024 + 1)
        state_before = client.get_manager_state()
        with pytest.raises(RuntimeError, match="Server error"):
            client.write(uuid, too_large)
        state_after = client.get_manager_state()
        assert state_after["free_blocks_count"] == state_before["free_blocks_count"]
        assert uuid not in client.get_manager_state().get("cached_items", [])

    def test_delete_and_read(self, client):
        uuid = _unique_uuid()
        client.write(uuid, b"temp data")
        state_before_delete = client.get_manager_state()
        client.delete(uuid)
        state_after_delete = client.get_manager_state()
        assert (
            state_after_delete["cached_items_count"]
            == state_before_delete["cached_items_count"] - 1
        )
        with pytest.raises(RuntimeError, match="Server error"):
            client.read(uuid)

    def test_pin_unpin(self, client):
        uuid = _unique_uuid()
        client.write(uuid, b"pinned item")
        state_before_pin = client.get_manager_state()
        client.pin(uuid)
        state_after_pin = client.get_manager_state()
        assert (
            state_after_pin["pinned_items_count"]
            == state_before_pin["pinned_items_count"] + 1
        )

        client.unpin(uuid)
        state_after_unpin = client.get_manager_state()
        assert (
            state_after_unpin["pinned_items_count"]
            == state_before_pin["pinned_items_count"]
        )

        result = client.read(uuid)
        assert result.tobytes() == b"pinned item"

        client.delete(uuid)


# ---------------------------------------------------------------------------
# Server metadata and state
# ---------------------------------------------------------------------------


class TestMetadata:
    """Check that server reports correct storage information."""

    def test_storage_info(self, client):
        info = client.get_storage_info()
        assert "name" in info
        assert info["size"] == 1024 * 1024
        assert info["block_size"] == 4096
        assert info["n_block"] == 256

    def test_manager_state_initial(self, client):
        state = client.get_manager_state()
        assert state["free_blocks_count"] == 256
        assert state["cached_items_count"] == 0
        assert state["pinned_items_count"] == 0
        assert state["total_items_count"] == 0
        assert state["writing_items_count"] == 0
        assert state["reading_items_count"] == 0
        assert state["idle_items_count"] == 0

    def test_manager_state_after_write(self, client):
        uuid = _unique_uuid()
        state_before = client.get_manager_state()
        client.write(uuid, b"state check")
        state_after = client.get_manager_state()
        needed = _blocks_needed(len(b"state check"))
        assert (
            state_after["cached_items_count"] == state_before["cached_items_count"] + 1
        )
        assert (
            state_after["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed
        )
        assert (
            state_after["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )
        client.delete(uuid)


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Limited concurrency test using threads (ZMQ sockets are thread‑safe)."""

    def test_concurrent_readers(self, client):
        import threading

        uuid = _unique_uuid()
        data = b"concurrent read data"
        client.write(uuid, data)

        results = []

        def reader():
            results.append(client.read(uuid).tobytes())

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == data for r in results)

        client.delete(uuid)

    def test_concurrent_writers(self, client):
        """Multiple writers with distinct UUIDs should succeed."""
        import threading

        uuids = [_unique_uuid() for _ in range(4)]
        datas = [f"writer-{i}".encode() for i in range(4)]

        state_before = client.get_manager_state()

        def writer(u, d):
            client.write(u, d)

        threads = [
            threading.Thread(target=writer, args=(u, d)) for u, d in zip(uuids, datas)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        state_after = client.get_manager_state()
        needed_blocks = sum(math.ceil(len(d) / 4096) for d in datas)
        assert state_after["cached_items_count"] == state_before[
            "cached_items_count"
        ] + len(uuids)
        assert (
            state_after["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed_blocks
        )
        assert (
            state_after["free_blocks_count"]
            == state_before["free_blocks_count"] - needed_blocks
        )

        # Verify data integrity and cleanup
        for u, d in zip(uuids, datas):
            result = client.read(u)
            assert result.tobytes() == d
            client.delete(u)
