# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import threading
import time

import numpy as np
import pytest
import torch

from vllm.multimodal.paged_shm.client import PagedShmClient
from vllm.multimodal.paged_shm.server import PagedShmServerProc
from vllm.multimodal.paged_shm.types import ShmWriteRequest
from vllm.utils import random_uuid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def server():
    """Start a single debug-enabled server for all tests."""
    server = PagedShmServerProc(size=1024 * 1024, block_size=4096, debug=True)
    server.start()
    yield server
    server.shutdown()


@pytest.fixture(scope="function")
def client(server):
    """
    Create a fresh client for each test and perform a debug cleanup
    before the test starts to remove any leftover waiters/tokens from
    previous tests.
    """
    c = PagedShmClient(address=server.address, pin=False)
    # Clean up any stale state from previous tests
    c.debug_cleanup()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_uuid() -> str:
    return f"test-{random_uuid()}"


def _blocks_needed(size: int, block_size: int = 4096) -> int:
    return math.ceil(size / block_size)


def _fill_memory_with_writing(client) -> str:
    """
    Allocate all free blocks with a writing item (open_write without close_write).
    This holds the blocks in the 'writing' state, preventing eviction.
    Returns the UUID so the caller can delete it to free space.
    """
    state = client.get_manager_state()
    free_blocks = state["free_blocks_count"]
    if free_blocks == 0:
        raise RuntimeError("No free blocks available")
    block_size = client.get_storage_info()["block_size"]
    size = free_blocks * block_size
    uuid = _unique_uuid()
    # Open write but do NOT close it, so blocks stay reserved
    client.open_write(
        [ShmWriteRequest(uuid=uuid, size=size, use_cache=True)], timeout=0.0
    )
    # Ensure we've consumed all free blocks
    new_state = client.get_manager_state()
    assert new_state["free_blocks_count"] == 0, "Failed to fill memory"
    return uuid


# ---------------------------------------------------------------------------
# Basic write / read
# ---------------------------------------------------------------------------


class TestWriteRead:
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
    def test_write_read_torch_gpu(self, server):
        client = PagedShmClient(address=server.address, pin=True)
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
# Large data
# ---------------------------------------------------------------------------


class TestMultiBlock:
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
# Context managers
# ---------------------------------------------------------------------------


class TestContextManagers:
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
        initial_free = state_before["free_blocks_count"]

        class TestException(Exception):
            pass

        with pytest.raises(TestException):  # noqa: SIM117
            with client.write_context(uuid, size) as ctx:
                client._storage.write(data, ctx.blocks)
                raise TestException("trigger rollback")

        state_after_rollback = client.get_manager_state()
        # With force delete, blocks are fully freed
        assert state_after_rollback["free_blocks_count"] == initial_free
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
    def test_read_nonexistent_uuid(self, client):
        with pytest.raises(RuntimeError, match="Server error"):
            client.read("nonexistent-uuid")

    def test_write_exceeding_block_count(self, client):
        uuid = _unique_uuid()
        info = client.get_storage_info()
        block_size = info["block_size"]
        free_blocks = client.get_manager_state()["free_blocks_count"]
        too_large = bytes((free_blocks + 1) * block_size)
        state_before = client.get_manager_state()
        with pytest.raises(RuntimeError, match="Server error"):
            client.write(uuid, too_large)
        state_after = client.get_manager_state()
        assert state_after["free_blocks_count"] == state_before["free_blocks_count"]
        assert state_after["total_items_count"] == state_before["total_items_count"]

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

    def test_delete_force_writing_item(self, client):
        """Delete can force-remove an item that is still being written."""
        uuid = _unique_uuid()
        # Allocate blocks but don't close write
        item = ShmWriteRequest(uuid=uuid, size=100, use_cache=True)
        client.open_write([item], timeout=0.0)

        # Delete should succeed despite ref_count == -1
        state_before = client.get_manager_state()
        client.delete(uuid)
        state_after = client.get_manager_state()
        # Blocks should be freed
        assert state_after["free_blocks_count"] == state_before["free_blocks_count"] + 1
        with pytest.raises(RuntimeError, match="Server error"):
            client.read(uuid)


# ---------------------------------------------------------------------------
# Server metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_storage_info(self, client):
        info = client.get_storage_info()
        assert "name" in info
        assert info["size"] == 1024 * 1024
        assert info["block_size"] == 4096
        assert info["n_block"] == 256

    def test_manager_state_initial(self, client):
        info = client.get_storage_info()
        n_block = info["n_block"]
        state = client.get_manager_state()
        assert state["free_blocks_count"] == n_block
        assert state["cached_items_count"] == 0
        assert state["total_items_count"] == 0
        assert state["writing_items_count"] == 0
        assert state["reading_items_count"] == 0
        assert state["idle_items_count"] == 0

    def test_manager_state_after_write(self, client):
        uuid = _unique_uuid()
        state_before = client.get_manager_state()
        data = b"state check"
        client.write(uuid, data)
        state_after = client.get_manager_state()
        needed = _blocks_needed(len(data))
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
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_readers(self, client):
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

        for u, d in zip(uuids, datas):
            result = client.read(u)
            assert result.tobytes() == d
            client.delete(u)


# ---------------------------------------------------------------------------
# Token protection tests (auto reference)
# ---------------------------------------------------------------------------


class TestTokenProtection:
    def test_auto_protection_holds_read_lock(self, client):
        """
        Writing with generate_read_token=True automatically reserves one read
        reference per token. Token can be open_read multiple times without being
        consumed until close_read is called.
        """
        uuid = _unique_uuid()
        data = b"auto protection test"
        state_before = client.get_manager_state()

        size, token = client.write(uuid, data, generate_read_token=True)

        state_after_write = client.get_manager_state()
        # Should have one extra reading item (the automatically reserved reference)
        assert (
            state_after_write["reading_items_count"]
            == state_before["reading_items_count"] + 1
        )
        # The item should NOT be cached yet (because it's held by the token)
        assert (
            state_after_write["cached_items_count"]
            == state_before["cached_items_count"]
        )
        needed = _blocks_needed(len(data))
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        # Token can be open_read multiple times without consuming it
        alloc1 = client.open_read(token, timeout=0.0)
        assert alloc1.size == len(data)
        state_after_read1 = client.get_manager_state()
        assert (
            state_after_read1["reading_items_count"]
            == state_after_write["reading_items_count"]
        )

        alloc2 = client.open_read(token, timeout=0.0)
        assert alloc2.size == len(data)
        state_after_read2 = client.get_manager_state()
        assert (
            state_after_read2["reading_items_count"]
            == state_after_write["reading_items_count"]
        )

        # Close the token to release the read reference and destroy it
        client.close_read(token)
        state_after_close = client.get_manager_state()
        assert (
            state_after_close["reading_items_count"]
            == state_before["reading_items_count"]
        )
        # Now item becomes idle and cacheable
        assert (
            state_after_close["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )

        # Token is destroyed; further opens should fail
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(token, timeout=0.0)

        client.delete(uuid)

    def test_token_allows_multiple_open_reads(self, client):
        """A token can be used for multiple open_read calls without consuming it."""
        uuid = _unique_uuid()
        data = b"multiple reads"
        size, token = client.write(uuid, data, generate_read_token=True)

        for _ in range(3):
            alloc = client.open_read(token, timeout=0.0)
            assert alloc.size == len(data)

        client.close_read(token)
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(token, timeout=0.0)

        client.delete(uuid)

    def test_auto_protection_with_async(self, client):
        """Async write with token automatically gets protection."""
        uuid = _unique_uuid()
        data = b"async auto protection"
        state_before = client.get_manager_state()

        size, future, token = client.write(
            uuid, data, generate_read_token=True, async_write=True
        )
        future.result()

        state_after_write = client.get_manager_state()
        assert (
            state_after_write["reading_items_count"]
            == state_before["reading_items_count"] + 1
        )
        assert (
            state_after_write["cached_items_count"]
            == state_before["cached_items_count"]
        )

        client.close_read(token)
        state_after_close = client.get_manager_state()
        assert (
            state_after_close["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )

        client.delete(uuid)


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_open_write_timeout_zero_raises_memory_error(self, client):
        filler_uuid = _fill_memory_with_writing(client)
        try:
            with pytest.raises(RuntimeError, match="Server error"):
                client.open_write(
                    [ShmWriteRequest(uuid=_unique_uuid(), size=100, use_cache=True)],
                    timeout=0.0,
                )
        finally:
            client.delete(filler_uuid)  # force-delete the writing item

    def test_open_write_timeout_positive_succeeds_after_space_freed(self, client):
        filler_uuid = _fill_memory_with_writing(client)

        small_uuid = _unique_uuid()
        result_holder = {}
        err_holder = {}

        def _waiter():
            try:
                alloc = client.open_write(
                    [ShmWriteRequest(uuid=small_uuid, size=100, use_cache=True)],
                    timeout=5.0,
                )
                result_holder["alloc"] = alloc
            except Exception as e:
                err_holder["err"] = e

        t = threading.Thread(target=_waiter)
        t.start()

        time.sleep(0.2)
        assert t.is_alive(), "open_write should be waiting, not finished"

        client.delete(filler_uuid)  # free space

        t.join(timeout=10.0)
        assert not t.is_alive(), "open_write did not complete in time"
        assert "err" not in err_holder, f"open_write failed: {err_holder.get('err')}"
        assert "alloc" in result_holder
        assert result_holder["alloc"][0].blocks

        client.close_write(small_uuid)
        client.delete(small_uuid)

    def test_open_write_timeout_negative_infinite(self, client):
        filler_uuid = _fill_memory_with_writing(client)

        small_uuid = _unique_uuid()
        result_holder = {}
        err_holder = {}

        def _waiter():
            try:
                alloc = client.open_write(
                    [ShmWriteRequest(uuid=small_uuid, size=100, use_cache=True)],
                    timeout=-1.0,
                )
                result_holder["alloc"] = alloc
            except Exception as e:
                err_holder["err"] = e

        t = threading.Thread(target=_waiter)
        t.start()

        time.sleep(0.2)
        assert t.is_alive(), "open_write should be waiting indefinitely"

        client.delete(filler_uuid)

        t.join(timeout=10.0)
        assert not t.is_alive(), "open_write did not complete after space freed"
        assert "err" not in err_holder, f"open_write failed: {err_holder.get('err')}"
        assert "alloc" in result_holder

        client.close_write(small_uuid)
        client.delete(small_uuid)

    def test_open_read_timeout_zero_raises_error(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        try:
            with pytest.raises(RuntimeError, match="Server error"):
                client.open_read(uuid, timeout=0.0)
        finally:
            client.close_write(uuid)
            client.delete(uuid)

    def test_open_read_timeout_positive_succeeds_after_close_write(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        result_holder = {}
        err_holder = {}

        def _waiter():
            try:
                item = client.open_read(uuid, timeout=5.0)
                result_holder["item"] = item
            except Exception as e:
                err_holder["err"] = e

        t = threading.Thread(target=_waiter)
        t.start()

        time.sleep(0.2)
        assert t.is_alive(), "open_read should be waiting, not finished"

        client.close_write(uuid)

        t.join(timeout=10.0)
        assert not t.is_alive(), "open_read did not complete in time"
        assert "err" not in err_holder, f"open_read failed: {err_holder.get('err')}"
        assert "item" in result_holder
        assert result_holder["item"].blocks

        client.close_read(uuid)
        client.delete(uuid)

    def test_open_read_timeout_negative_infinite(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        result_holder = {}
        err_holder = {}

        def _waiter():
            try:
                item = client.open_read(uuid, timeout=-1.0)
                result_holder["item"] = item
            except Exception as e:
                err_holder["err"] = e

        t = threading.Thread(target=_waiter)
        t.start()

        time.sleep(0.2)
        assert t.is_alive(), "open_read should be waiting indefinitely"

        client.close_write(uuid)

        t.join(timeout=10.0)
        assert not t.is_alive(), "open_read did not complete after close_write"
        assert "err" not in err_holder, f"open_read failed: {err_holder.get('err')}"
        assert "item" in result_holder

        client.close_read(uuid)
        client.delete(uuid)

    def test_open_write_timeout_expires(self, client):
        filler_uuid = _fill_memory_with_writing(client)
        small_uuid = _unique_uuid()
        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_write(
                [ShmWriteRequest(uuid=small_uuid, size=100, use_cache=True)],
                timeout=0.5,
            )
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.4, f"Timeout should be ~0.5s, got {elapsed:.2f}s"
        client.delete(filler_uuid)

    def test_open_read_timeout_expires(self, client):
        uuid = _unique_uuid()
        # Open write without closing, so item stays in writing state
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )
        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(uuid, timeout=0.5)
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.4, f"Timeout should be ~0.5s, got {elapsed:.2f}s"
        # Clean up
        client.close_write(uuid)
        client.delete(uuid)

    def test_wait_write_timeout_expires(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )
        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="Server error"):
            client.wait_write(uuid, timeout=0.5)
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.4
        client.close_write(uuid)
        client.delete(uuid)


# ---------------------------------------------------------------------------
# Asynchronous writes
# ---------------------------------------------------------------------------


class TestAsyncWrite:
    def test_async_write_basic(self, client):
        uuid = _unique_uuid()
        data = b"async test data"
        state_before = client.get_manager_state()

        size, future, _ = client.write(uuid, data, async_write=True)
        assert size == len(data)
        future.result()

        state_after = client.get_manager_state()
        needed = _blocks_needed(len(data))
        assert (
            state_after["cached_items_count"] == state_before["cached_items_count"] + 1
        )
        assert (
            state_after["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = client.read(uuid)
        assert result.tobytes() == data
        client.delete(uuid)

    def test_async_write_with_token_auto_protection(self, client):
        """Async write with token automatically holds a reference."""
        uuid = _unique_uuid()
        data = b"async token protection"
        state_before = client.get_manager_state()

        size, future, token = client.write(
            uuid, data, generate_read_token=True, async_write=True
        )
        assert size == len(data)
        future.result()

        state_after_write = client.get_manager_state()
        assert (
            state_after_write["reading_items_count"]
            == state_before["reading_items_count"] + 1
        )
        assert (
            state_after_write["cached_items_count"]
            == state_before["cached_items_count"]
        )

        # Can open_read multiple times via token without consuming it
        alloc1 = client.open_read(token, timeout=0.0)
        assert alloc1.size == len(data)
        alloc2 = client.open_read(token, timeout=0.0)
        assert alloc2.size == len(data)

        client.close_read(token)
        state_after_close = client.get_manager_state()
        assert (
            state_after_close["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )
        client.delete(uuid)

    def test_async_write_exception_rollback(self, client):
        uuid = _unique_uuid()
        invalid_array = np.array([{1: 1}, {2: 2}])

        size, future, _ = client.write(uuid, invalid_array, async_write=True)
        with pytest.raises(TypeError):
            future.result()

        with pytest.raises(RuntimeError, match="Server error"):
            client.read(uuid)

    def test_async_write_concurrent(self, client):
        uuids = [_unique_uuid() for _ in range(4)]
        datas = [f"async-{i}".encode() for i in range(4)]
        futures = []

        for u, d in zip(uuids, datas):
            _, fut, _ = client.write(u, d, async_write=True)
            futures.append(fut)

        for fut in futures:
            fut.result()

        for u, d in zip(uuids, datas):
            result = client.read(u)
            assert result.tobytes() == d
            client.delete(u)


# ---------------------------------------------------------------------------
# Pre‑allocated blocks
# ---------------------------------------------------------------------------


class TestPreAllocatedBlocks:
    def test_write_with_preallocated_blocks(self, client):
        uuid = _unique_uuid()
        data = b"preallocated write"
        size = len(data)

        item = ShmWriteRequest(uuid=uuid, size=size, use_cache=True)
        alloc = client.open_write([item], timeout=0.0)
        blocks = alloc[0].blocks

        client.write(uuid, data, blocks=blocks)
        result = client.read(uuid)
        assert result.tobytes() == data
        client.delete(uuid)

    def test_read_with_preallocated_blocks(self, client):
        uuid = _unique_uuid()
        data = b"preallocated read"
        client.write(uuid, data)

        item = client.open_read(uuid, timeout=0.0)
        blocks = item.blocks
        size = item.size

        result = client._storage.read_to_numpy(size, blocks)
        assert result.tobytes() == data

        client.close_read(uuid)
        client.delete(uuid)


# ---------------------------------------------------------------------------
# wait_write
# ---------------------------------------------------------------------------


class TestWaitWrite:
    def test_wait_write_immediate_when_readable(self, client):
        uuid = _unique_uuid()
        data = b"test data"
        client.write(uuid, data)
        client.wait_write(uuid, timeout=0.0)
        client.delete(uuid)

    def test_wait_write_blocks_until_close_write(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        def _close_writer():
            time.sleep(0.3)
            client.close_write(uuid)

        close_thread = threading.Thread(target=_close_writer)
        close_thread.start()

        client.wait_write(uuid, timeout=5.0)
        close_thread.join()
        client.delete(uuid)

    def test_wait_write_timeout_zero_raises(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )
        try:
            with pytest.raises(RuntimeError, match="Server error"):
                client.wait_write(uuid, timeout=0.0)
        finally:
            client.close_write(uuid)
            client.delete(uuid)

    def test_wait_write_timeout_positive_succeeds_after_close_write(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        def _close_writer():
            time.sleep(0.3)
            client.close_write(uuid)

        close_thread = threading.Thread(target=_close_writer)
        close_thread.start()

        client.wait_write(uuid, timeout=5.0)
        close_thread.join()
        client.delete(uuid)

    def test_wait_write_timeout_negative_infinite(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        def _close_writer():
            time.sleep(0.3)
            client.close_write(uuid)

        close_thread = threading.Thread(target=_close_writer)
        close_thread.start()

        client.wait_write(uuid, timeout=-1.0)
        close_thread.join()
        client.delete(uuid)

    def test_wait_write_multiple_waiters(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        num_waiters = 3
        done_count = 0
        errors = []

        def _waiter():
            nonlocal done_count
            try:
                client.wait_write(uuid, timeout=5.0)
                done_count += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_waiter) for _ in range(num_waiters)]
        for t in threads:
            t.start()

        time.sleep(0.2)
        client.close_write(uuid)

        for t in threads:
            t.join(timeout=5.0)
        assert not any(t.is_alive() for t in threads)
        assert len(errors) == 0
        assert done_count == num_waiters
        client.delete(uuid)

    def test_wait_write_nonexistent_uuid(self, client):
        uuid = _unique_uuid()
        with pytest.raises(RuntimeError, match="Server error"):
            client.wait_write(uuid, timeout=0.0)

    def test_wait_write_then_open_read(self, client):
        uuid = _unique_uuid()
        data = b"test wait then read"
        size = len(data)

        alloc = client.open_write(
            [ShmWriteRequest(uuid=uuid, size=size, use_cache=True)], timeout=0.0
        )
        blocks = alloc[0].blocks

        def _write_and_close():
            client._storage.write(data, blocks)
            client.close_write(uuid)

        write_thread = threading.Thread(target=_write_and_close)
        write_thread.start()

        client.wait_write(uuid, timeout=5.0)
        result = client.read(uuid)
        assert result.tobytes() == data

        write_thread.join()
        client.delete(uuid)


# ---------------------------------------------------------------------------
# Read Token tests
# ---------------------------------------------------------------------------


class TestReadToken:
    def test_generate_and_use_token_sync(self, client):
        uuid = _unique_uuid()
        data = b"token test data"

        size, token = client.write(uuid, data, generate_read_token=True)
        assert token is not None
        assert size == len(data)

        # Token can be open_read multiple times without consuming it
        alloc1 = client.open_read(token, timeout=0.0)
        assert alloc1.size == len(data)
        alloc2 = client.open_read(token, timeout=0.0)
        assert alloc2.size == len(data)

        client.close_read(token)
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(token, timeout=0.0)

        # The original UUID still works (no read lock held)
        result = client.read(uuid)
        assert result.tobytes() == data
        client.delete(uuid)

    def test_generate_and_use_token_async(self, client):
        uuid = _unique_uuid()
        data = b"async token"

        size, future, token = client.write(
            uuid, data, generate_read_token=True, async_write=True
        )
        future.result()

        alloc1 = client.open_read(token, timeout=0.0)
        assert alloc1.size == len(data)
        alloc2 = client.open_read(token, timeout=0.0)
        assert alloc2.size == len(data)

        client.close_read(token)
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(token, timeout=0.0)

        client.delete(uuid)

    def test_token_with_wait_write(self, client):
        uuid = _unique_uuid()
        data = b"token+wait test"
        size = len(data)
        item = ShmWriteRequest(uuid=uuid, size=size, generate_read_token=True)
        alloc = client.open_write([item], timeout=0.0)
        token = alloc[0].read_token
        assert token is not None
        blocks = alloc[0].blocks

        def _write_and_close():
            client._storage.write(data, blocks)
            client.close_write(uuid)

        t = threading.Thread(target=_write_and_close)
        t.start()

        client.wait_write(token, timeout=5.0)

        alloc1 = client.open_read(token, timeout=0.0)
        assert alloc1.size == len(data)
        alloc2 = client.open_read(token, timeout=0.0)
        assert alloc2.size == len(data)

        client.close_read(token)
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(token, timeout=0.0)

        t.join()
        client.delete(uuid)

    def test_token_cannot_be_reused_after_close_read(self, client):
        uuid = _unique_uuid()
        data = b"reuse test"
        size, token = client.write(uuid, data, generate_read_token=True)

        alloc = client.open_read(token, timeout=0.0)
        assert alloc.size == len(data)

        client.close_read(token)
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(token, timeout=0.0)

        client.delete(uuid)

    def test_token_with_read_context(self, client):
        uuid = _unique_uuid()
        data = b"context token"
        size, token = client.write(uuid, data, generate_read_token=True)

        with client.read_context(token) as ctx:
            result = client._storage.read_to_numpy(ctx.size, ctx.blocks)
            assert result.tobytes() == data

        # After context exit, token is destroyed
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(token, timeout=0.0)

        client.delete(uuid)

    def test_token_info_and_wait_write_do_not_consume(self, client):
        uuid = _unique_uuid()
        data = b"dummy"
        size = len(data)
        item = ShmWriteRequest(uuid=uuid, size=size, generate_read_token=True)
        alloc = client.open_write([item], timeout=0.0)
        token = alloc[0].read_token
        blocks = alloc[0].blocks

        info = client.get_info(token)
        assert info["uuid"] == uuid

        def _write_and_close():
            client._storage.write(data, blocks)
            client.close_write(uuid)

        t = threading.Thread(target=_write_and_close)
        t.start()

        client.wait_write(token, timeout=5.0)
        t.join()

        alloc = client.open_read(token, timeout=0.0)
        assert alloc.size == len(data)

        client.close_read(token)
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(token, timeout=0.0)

        client.delete(uuid)

    def test_token_cleanup_on_delete(self, client):
        uuid = _unique_uuid()
        data = b"delete token"
        size, token = client.write(uuid, data, generate_read_token=True)

        # Delete forces removal, ignoring the token reference
        client.delete(uuid)

        # Token is gone
        with pytest.raises(RuntimeError, match="Server error"):
            client.open_read(uuid, timeout=0.0)

        # close_read on token should fail because token no longer exists
        with pytest.raises(RuntimeError, match="not found"):
            client.close_read(token)


# ---------------------------------------------------------------------------
# Info and cache behavior
# ---------------------------------------------------------------------------


class TestInfoAndCache:
    def test_get_info(self, client):
        uuid = _unique_uuid()
        data = b"info test"
        client.write(uuid, data)

        info = client.get_info(uuid)
        assert info["uuid"] == uuid
        assert info["size"] == len(data)
        assert info["ref_count"] == 0

        with client.read_context(uuid):
            info_reading = client.get_info(uuid)
            assert info_reading["ref_count"] == 1

        client.delete(uuid)

    def test_use_cache_false_does_not_cache(self, client):
        """
        Behavior of non-cacheable items (use_cache=False):
        - If no read reference is retained after close_write,
          the item is deleted immediately.
        - If a read reference is retained via generate_read_token=True,
          the item survives until the token is closed, then it is deleted.
        """
        uuid = _unique_uuid()
        data = b"non-cacheable"

        # Scenario 1: no read reference retained -> delete immediately after write
        client.write(uuid, data, use_cache=False)
        with pytest.raises(RuntimeError, match="Server error"):
            client.read(uuid)

        # Scenario 2: retain via token, item is readable, auto-deleted after close_read
        uuid2 = _unique_uuid()
        size, token = client.write(
            uuid2, data, use_cache=False, generate_read_token=True
        )
        assert token is not None

        # Reading via token succeeds (token holds a reference)
        client.open_read(token)

        # Close the read reference (destroy token) -> item should be deleted
        client.close_read(token)

        # Now reading via UUID should fail (item no longer exists)
        with pytest.raises(RuntimeError, match="Server error"):
            client.read(uuid2)

    def test_use_cache_true_caches_after_close(self, client):
        uuid = _unique_uuid()
        data = b"cacheable"
        state_before = client.get_manager_state()

        client.write(uuid, data, use_cache=True)

        state_after = client.get_manager_state()
        assert (
            state_after["cached_items_count"] == state_before["cached_items_count"] + 1
        )
        assert state_after["idle_items_count"] == state_before["idle_items_count"] + 1

        client.delete(uuid)


# ---------------------------------------------------------------------------
# Multi-client scenarios
# ---------------------------------------------------------------------------


class TestMultiClient:
    def test_two_clients_read_write(self, server):
        client1 = PagedShmClient(address=server.address, pin=False)
        client2 = PagedShmClient(address=server.address, pin=False)
        try:
            uuid = _unique_uuid()
            data = b"multi-client test"

            client1.write(uuid, data)
            result = client2.read(uuid)
            assert result.tobytes() == data

            client1.delete(uuid)
            with pytest.raises(RuntimeError, match="Server error"):
                client2.read(uuid)

        finally:
            client1.close()
            client2.close()


# ---------------------------------------------------------------------------
# Delete with pending waiters
# ---------------------------------------------------------------------------


class TestDeleteWithWaiters:
    def test_delete_clears_pending_open_reads(self, client):
        uuid = _unique_uuid()
        # Start a write but don't close it
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        # Start a thread waiting for open_read
        result_holder = {}
        err_holder = {}

        def waiter():
            try:
                client.open_read(uuid, timeout=5.0)
                result_holder["done"] = True
            except Exception as e:
                err_holder["err"] = e

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)  # ensure waiter is queued

        # Delete should force removal and wake waiters with error
        client.delete(uuid)  # succeeds, forces cleanup

        t.join(timeout=2.0)
        assert not t.is_alive()
        assert "err" in err_holder
        assert "deleted" in str(err_holder["err"]).lower()

    def test_delete_clears_pending_wait_write(self, client):
        uuid = _unique_uuid()
        client.open_write(
            [ShmWriteRequest(uuid=uuid, size=100, use_cache=True)], timeout=0.0
        )

        result_holder = {}
        err_holder = {}

        def waiter():
            try:
                client.wait_write(uuid, timeout=5.0)
                result_holder["done"] = True
            except Exception as e:
                err_holder["err"] = e

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)

        client.delete(uuid)  # force delete

        t.join(timeout=2.0)
        assert not t.is_alive()
        assert "err" in err_holder
        assert "deleted" in str(err_holder["err"]).lower()
