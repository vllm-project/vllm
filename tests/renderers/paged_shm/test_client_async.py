# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
import multiprocessing as mp

import numpy as np
import pytest
import pytest_asyncio
import torch

from vllm.renderers.paged_shm.client_async import AsyncPagedShmClient
from vllm.renderers.paged_shm.server import zmq_server
from vllm.utils import random_uuid

# ---------------------------------------------------------------------------
# Fixtures (sync server, async client)
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


@pytest_asyncio.fixture
async def async_client(server_address):
    """
    Create an AsyncPagedShmClient connected to the test server.
    Memory pinning is disabled by default; GPU tests must create their own
    client with ``pin=True``.
    """
    client = AsyncPagedShmClient(address=server_address, pin=False)
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_uuid() -> str:
    return f"test-{random_uuid()}"


def _blocks_needed(size: int, block_size: int = 4096) -> int:
    return math.ceil(size / block_size)


# ---------------------------------------------------------------------------
# Basic write / read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWriteRead:
    async def test_write_read_bytes(self, async_client):
        uuid = _unique_uuid()
        data = b"Hello, async shared memory!"
        state_before = await async_client.get_manager_state()

        await async_client.write(uuid, data)

        state_after_write = await async_client.get_manager_state()
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

        result = await async_client.read(uuid)
        assert isinstance(result, np.ndarray)
        assert result.tobytes() == data

        await async_client.delete(uuid)
        state_final = await async_client.get_manager_state()
        assert state_final["cached_items_count"] == state_before["cached_items_count"]
        assert state_final["free_blocks_count"] == state_before["free_blocks_count"]

    async def test_write_read_numpy(self, async_client):
        uuid = _unique_uuid()
        original = np.arange(100, dtype=np.float32)
        state_before = await async_client.get_manager_state()

        await async_client.write(uuid, original)

        state_after_write = await async_client.get_manager_state()
        needed = _blocks_needed(original.nbytes)
        assert (
            state_after_write["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = await async_client.read(uuid)
        np.testing.assert_array_equal(result.view(np.float32), original)

        await async_client.delete(uuid)

    async def test_write_read_torch_cpu(self, async_client):
        uuid = _unique_uuid()
        original = torch.arange(50, dtype=torch.int32)
        state_before = await async_client.get_manager_state()

        await async_client.write(uuid, original)

        state_after_write = await async_client.get_manager_state()
        needed = _blocks_needed(original.numel() * original.element_size())
        assert (
            state_after_write["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result_np = await async_client.read(uuid)
        assert isinstance(result_np, np.ndarray)
        result = torch.from_numpy(result_np)
        torch.testing.assert_close(result.view(torch.int32)[: len(original)], original)

        await async_client.delete(uuid)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    async def test_write_read_torch_gpu(self, server_address):
        async_client = AsyncPagedShmClient(address=server_address, pin=True)
        try:
            uuid = _unique_uuid()
            original = torch.randint(0, 255, (500,), dtype=torch.uint8, device="cuda")
            state_before = await async_client.get_manager_state()

            await async_client.write(uuid, original)

            state_after_write = await async_client.get_manager_state()
            needed = _blocks_needed(original.numel())
            assert (
                state_after_write["cached_items_count"]
                == state_before["cached_items_count"] + 1
            )
            assert (
                state_after_write["free_blocks_count"]
                == state_before["free_blocks_count"] - needed
            )

            result = await async_client.read(uuid, device="cuda")
            assert isinstance(result, torch.Tensor)
            torch.testing.assert_close(result, original)

            await async_client.delete(uuid)
        finally:
            await async_client.close()


# ---------------------------------------------------------------------------
# Large data (multi‑block)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMultiBlock:
    @pytest.mark.parametrize("size", [8000, 16384, 20000])
    async def test_bytes_multi_block(self, async_client, size):
        uuid = _unique_uuid()
        data = bytes(np.random.bytes(size))
        state_before = await async_client.get_manager_state()

        await async_client.write(uuid, data)

        state_after_write = await async_client.get_manager_state()
        needed = _blocks_needed(size)
        assert (
            state_after_write["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = await async_client.read(uuid)
        assert result.tobytes() == data

        await async_client.delete(uuid)
        state_final = await async_client.get_manager_state()
        assert state_final["free_blocks_count"] == state_before["free_blocks_count"]

    @pytest.mark.parametrize("size", [8000, 16384, 20000])
    async def test_numpy_multi_block(self, async_client, size):
        uuid = _unique_uuid()
        original = np.random.randint(0, 256, size, dtype=np.uint8)
        state_before = await async_client.get_manager_state()

        await async_client.write(uuid, original)

        state_after_write = await async_client.get_manager_state()
        needed = _blocks_needed(original.nbytes)
        assert (
            state_after_write["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = await async_client.read(uuid)
        np.testing.assert_array_equal(result, original)

        await async_client.delete(uuid)

    @pytest.mark.parametrize("size", [8000, 16384, 20000])
    async def test_torch_multi_block(self, async_client, size):
        uuid = _unique_uuid()
        original = torch.randint(0, 256, (size,), dtype=torch.uint8)
        state_before = await async_client.get_manager_state()

        await async_client.write(uuid, original)

        state_after_write = await async_client.get_manager_state()
        needed = _blocks_needed(size)
        assert (
            state_after_write["cached_blocks_count"]
            == state_before["cached_blocks_count"] + needed
        )
        assert (
            state_after_write["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result_np = await async_client.read(uuid)
        result = torch.from_numpy(result_np)
        torch.testing.assert_close(result, original)

        await async_client.delete(uuid)


# ---------------------------------------------------------------------------
# Context managers (async with)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncContextManagers:
    async def test_write_context_commit(self, async_client):
        uuid = _unique_uuid()
        data = b"async context write"
        size = len(data)
        state_before = await async_client.get_manager_state()

        async with async_client.write_context(uuid, size) as ctx:
            async_client._storage.write(data, ctx.blocks)

        state_after_commit = await async_client.get_manager_state()
        needed = _blocks_needed(size)
        assert (
            state_after_commit["cached_items_count"]
            == state_before["cached_items_count"] + 1
        )
        assert (
            state_after_commit["free_blocks_count"]
            == state_before["free_blocks_count"] - needed
        )

        result = await async_client.read(uuid)
        assert result.tobytes() == data

        await async_client.delete(uuid)

    async def test_write_context_rollback(self, async_client):
        uuid = _unique_uuid()
        data = b"should not be visible"
        size = len(data)
        state_before = await async_client.get_manager_state()

        class TestException(Exception):
            pass

        with pytest.raises(TestException):
            async with async_client.write_context(uuid, size) as ctx:
                async_client._storage.write(data, ctx.blocks)
                raise TestException("trigger rollback")

        state_after_rollback = await async_client.get_manager_state()
        assert (
            state_after_rollback["free_blocks_count"]
            == state_before["free_blocks_count"]
        )
        assert (
            state_after_rollback["cached_items_count"]
            == state_before["cached_items_count"]
        )

        with pytest.raises(RuntimeError, match="Server error"):
            await async_client.read(uuid)

    async def test_read_context(self, async_client):
        uuid = _unique_uuid()
        data = b"read context test"
        await async_client.write(uuid, data)

        state_before_read = await async_client.get_manager_state()
        reading_before = state_before_read["reading_items_count"]

        async with async_client.read_context(uuid) as ctx:
            state_during = await async_client.get_manager_state()
            assert state_during["reading_items_count"] == reading_before + 1
            result = async_client._storage.read_to_numpy(ctx.size, ctx.blocks)
            assert result.tobytes() == data

        state_after = await async_client.get_manager_state()
        assert state_after["reading_items_count"] == reading_before

        await async_client.delete(uuid)

    async def test_iterator_numpy_context(self, async_client):
        uuid = _unique_uuid()
        original = np.random.randint(0, 256, 10000, dtype=np.uint8)
        await async_client.write(uuid, original)

        async with async_client.get_iterator_numpy(uuid, len(original)) as it:
            blocks = []
            for arr, valid_len in it:
                blocks.append(arr[:valid_len])
        assembled = np.concatenate(blocks)
        np.testing.assert_array_equal(assembled, original)

        await async_client.delete(uuid)

    async def test_iterator_tensor_context(self, async_client):
        uuid = _unique_uuid()
        original = torch.randint(0, 256, (10000,), dtype=torch.uint8)
        await async_client.write(uuid, original)

        async with async_client.get_iterator_tensor(uuid, len(original)) as it:
            blocks = []
            for tensor, valid_len in it:
                blocks.append(tensor[:valid_len])
        assembled = torch.cat(blocks)
        torch.testing.assert_close(assembled, original)

        await async_client.delete(uuid)


# ---------------------------------------------------------------------------
# Error and edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrors:
    async def test_read_nonexistent_uuid(self, async_client):
        with pytest.raises(RuntimeError, match="Server error"):
            await async_client.read("nonexistent-uuid")

    async def test_write_exceeding_block_count(self, async_client):
        uuid = _unique_uuid()
        too_large = bytes(1024 * 1024 + 1)
        state_before = await async_client.get_manager_state()
        with pytest.raises(RuntimeError, match="Server error"):
            await async_client.write(uuid, too_large)
        state_after = await async_client.get_manager_state()
        assert state_after["free_blocks_count"] == state_before["free_blocks_count"]

    async def test_read_request_exceeding_data_size(self, async_client):
        uuid = _unique_uuid()
        data = b"short"
        await async_client.write(uuid, data)
        with pytest.raises(ValueError, match="exceeds available data size"):
            async with async_client.get_iterator_tensor(uuid, 100):
                pass
        await async_client.delete(uuid)

    async def test_delete_and_read(self, async_client):
        uuid = _unique_uuid()
        await async_client.write(uuid, b"temp data")
        state_before_delete = await async_client.get_manager_state()
        await async_client.delete(uuid)
        state_after_delete = await async_client.get_manager_state()
        assert (
            state_after_delete["cached_items_count"]
            == state_before_delete["cached_items_count"] - 1
        )
        with pytest.raises(RuntimeError, match="Server error"):
            await async_client.read(uuid)

    async def test_pin_unpin(self, async_client):
        uuid = _unique_uuid()
        await async_client.write(uuid, b"pinned item")
        state_before_pin = await async_client.get_manager_state()
        await async_client.pin(uuid)
        state_after_pin = await async_client.get_manager_state()
        assert (
            state_after_pin["pinned_items_count"]
            == state_before_pin["pinned_items_count"] + 1
        )

        await async_client.unpin(uuid)
        state_after_unpin = await async_client.get_manager_state()
        assert (
            state_after_unpin["pinned_items_count"]
            == state_before_pin["pinned_items_count"]
        )

        result = await async_client.read(uuid)
        assert result.tobytes() == b"pinned item"
        await async_client.delete(uuid)


# ---------------------------------------------------------------------------
# Server metadata and state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMetadata:
    async def test_storage_info(self, async_client):
        info = await async_client.get_storage_info()
        assert info["size"] == 1024 * 1024
        assert info["block_size"] == 4096
        assert info["n_block"] == 256

    async def test_manager_state_initial(self, async_client):
        state = await async_client.get_manager_state()
        assert state["free_blocks_count"] == 256
        assert state["cached_items_count"] == 0
        assert state["pinned_items_count"] == 0
        assert state["total_items_count"] == 0
        assert state["writing_items_count"] == 0
        assert state["reading_items_count"] == 0
        assert state["idle_items_count"] == 0

    async def test_manager_state_after_write(self, async_client):
        uuid = _unique_uuid()
        state_before = await async_client.get_manager_state()
        await async_client.write(uuid, b"state check")
        state_after = await async_client.get_manager_state()
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
        await async_client.delete(uuid)


# ---------------------------------------------------------------------------
# Concurrent access (asyncio tasks)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncConcurrency:
    async def test_concurrent_readers(self, async_client):
        uuid = _unique_uuid()
        data = b"concurrent async read"
        await async_client.write(uuid, data)

        async def reader():
            result = await async_client.read(uuid)
            return result.tobytes()

        tasks = [asyncio.create_task(reader()) for _ in range(4)]
        results = await asyncio.gather(*tasks)
        assert all(r == data for r in results)

        await async_client.delete(uuid)

    async def test_concurrent_writers(self, async_client):
        """Multiple writers with distinct UUIDs should succeed."""
        uuids = [_unique_uuid() for _ in range(4)]
        datas = [f"writer-{i}".encode() for i in range(4)]

        state_before = await async_client.get_manager_state()

        async def writer(u, d):
            await async_client.write(u, d)

        tasks = [asyncio.create_task(writer(u, d)) for u, d in zip(uuids, datas)]
        await asyncio.gather(*tasks)

        state_after = await async_client.get_manager_state()
        needed_blocks = sum(_blocks_needed(len(d)) for d in datas)
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

        # Verify data integrity
        for u, d in zip(uuids, datas):
            result = await async_client.read(u)
            assert result.tobytes() == d
            await async_client.delete(u)
